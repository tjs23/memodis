import os, sys
import numpy as np


from collections import defaultdict
#from matplotlib.pyplot import imread, imsave
from scipy.misc import imread, imsave

PROG_NAME = 'memodis'
VERSION = '1.0.0'
DESCRIPTION = 'A simple cellular membrane molecule diffusion simulator.'

DFLT_N_SAMPLES = 100
DFLT_N_STEPS = 10000
DFLT_TIMESTEP = 0.05
DFLT_D_BOUND = 0.01
DFLT_D_FREE = 25.0
DFLT_INIT_STEPS = 1000
DFLT_K_ON = 0.2
DFLT_ALT_K_ON = 0.2 
DFLT_K_OFF = 0.15
DFLT_ALT_K_OFF = 0.1
DFLT_MAX_OCC = 1

GRAD_STOPS = [0.0, 0.25, 0.5, 0.75, 1.0]
GRAD_RED   = [0.0, 0.2,  0.5, 0.0,  1.0]
GRAD_GREEN = [0.0, 0.2,  0.0, 1.0,  1.0]
GRAD_BLUE  = [0.0, 0.2,  0.5, 0.0,  0.0]  

IMG_FILTER = [[0.5, 1.0, 0.5],
              [1.0, 1.0, 1.0],
              [0.5, 1.0, 0.5]]

NEWLINE_CHARS = 0


def report(msg, line_return):
 
  global NEWLINE_CHARS

  if line_return:
    fmt = '\r%%-%ds' % max(NEWLINE_CHARS, len(msg))
    sys.stdout.write(fmt % msg)
    sys.stdout.flush()
    NEWLINE_CHARS = len(msg)
    
  else: 
    if NEWLINE_CHARS:
      print('')
      
    print(msg)
    NEWLINE_CHARS = 0


def warn(msg, prefix='WARNING', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)

 
def critical(msg, prefix='FAILURE', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)
  sys.exit(0)


def info(msg, prefix='INFO', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)

  
def check_invalid_file(file_path):
  
  msg = ''
  
  if not os.path.exists(file_path):
    msg = 'File "%s" does not exist' % file_path
 
  elif not os.path.isfile(file_path):
    msg = 'Location "%s" is not a regular file' % file_path
  
  elif os.stat(file_path).st_size == 0:
    msg = 'File "%s" is of zero size ' % file_path
    
  elif not os.access(file_path, os.R_OK):
    msg = 'File "%s" is not readable' % file_path

  if msg:
    critical(msg)
  
  return msg


def test_imports():
  import sys
  from distutils.core import run_setup
  
  critical = False

  try:
    from scipy import ndimage  
    
  except ImportError as err:
    critical = True
    warn('Critical Python module "scipy" is not installed or accessible')
  
  try:
    import numpy
    
  except ImportError as err:
    critical = True
    warn('Critical Python module "numpy" is not installed or accessible')

  try:
    import cython
  except ImportError as err:
    critical = True
    warn('Critical Python module "cython" is not installed or accessible')
  
  try:
    import dyn_sim
  
  except ImportError as err:
    warn('Utility C/Cython code not compiled. Attempting to compile now...')    
    run_setup('setup_cython.py', ['build_ext', '--inplace'])
    
    try:
      import dyn_sim
      warn('MEMODIS C/Cython code compiled. Please re-run command.')
      sys.exit(0)
      
    except ImportError as err:
      critical = True
      warn('Utility C/Cython code compilation/import failed')   
    
  if critical:
    critical('MEMODIS cannot proceed because critical Python modules are not available')
    
    
def _read_roi_file(file_path):
  
  rois = []
  
  with open(file_path) as file_obj:
    for line in file_obj:
      if line[0] == '#':
        continue
      
      line = line.strip()
      data = line.split()
      
      if not data:
        continue
        
      if len(data) != 5:
        critical('Region of interest lines must have 5 witespace-separate columns: x_pos, y_pos, radius, num_mem, group')
      
      try:
        vals = [float(x) for x in data[0:4]] + [data[4]]
                
      except ValueError:
        critical('Region of interest line "%s" values not interpretable' % line)
      
      rois.append(vals)

  if not rois:
    warn('Regions file did not specify any regions.')
  
  return rois

def run_dynamics(scene_pixmap, coords, out_img_dir, monitor_regions,
                 n_samples, n_steps, timestep, d_bound, d_free,
                 init_steps, k_on, alt_k_on, k_off, alt_k_off, max_occ,
                 pixel_decay=0, pixel_width=2, free_color=(255, 0, 0),
                 bound_color=(255, 255, 0)):
  
  from dyn_sim import dynamics_steps, add_pixmap_particles

  n_coords = len(coords)
  
  bg_pixmap = np.array(scene_pixmap)
  
  origin_mask = (bg_pixmap[:,:,0] > 0) & (bg_pixmap[:,:,1] == 0) & (bg_pixmap[:,:,2] == 0)
  bg_pixmap[origin_mask] = (0, 0, 0, 255)
  
  nz = bg_pixmap[:,:,:3].sum(axis=2) > 0
  bg_pixmap[nz] = (64, 64, 64, 255)
  pixmap = np.array(bg_pixmap)
  colors = np.array(free_color+bound_color, np.uint8)
  
  if out_img_dir:
    add_pixmap_particles(pixmap, bg_pixmap, coords, colors, pixel_decay, pixel_width)
    file_path = os.path.join(out_img_dir, 'start.png')
    imsave(file_path, pixmap)
  
  # Adapt parameters
  # - units of seconds converted to timestep probabilities etc.
  # - diffusion represented by Gaussian sigma
  
  sigma_free = (d_free * timestep * 2.0) ** 0.5
  sigma_bound = (d_bound * timestep * 2.0) ** 0.5
  
  p_bind = k_on * timestep
  p_unbind = k_off * timestep
  
  alt_p_bind = alt_k_on * timestep
  alt_p_unbind = alt_k_off * timestep

  local_steps = max(1, n_steps/n_samples)
  # monitor_regions [(x_pos, y_pos, radius),]  
  
  # initial fast diffusion for faster convergence
  info('  .. pre-equilibriation with %d steps' % init_steps, line_return=True)
  particle_params = np.array([[10.0*sigma_free, 10.0*sigma_bound, p_bind, p_unbind, alt_p_bind, alt_p_unbind]] * n_coords) 
  coords, region_counts = dynamics_steps(scene_pixmap, coords, particle_params,
                                         monitor_regions, init_steps, max_occ)
                                         
  if out_img_dir:
    add_pixmap_particles(pixmap, bg_pixmap, coords, colors, pixel_decay, pixel_width)
    file_path = os.path.join(out_img_dir, 'pre-equib.png')
    imsave(file_path, pixmap)

  # Different particles could potentially be modelled differently
  info('  .. equilibriation with %d steps' % init_steps)
  particle_params = np.array([[sigma_free, sigma_bound, p_bind, p_unbind, alt_p_bind, alt_p_unbind]] * n_coords) 
  coords, region_counts = dynamics_steps(scene_pixmap, coords, particle_params,
                                         monitor_regions, init_steps, max_occ)
  
  if out_img_dir:
    add_pixmap_particles(pixmap, bg_pixmap, coords, colors, pixel_decay, pixel_width)
    file_path = os.path.join(out_img_dir, 'equib.png')
    imsave(file_path, pixmap)

  count_data = []
  n, m, dims = pixmap.shape
  density = np.zeros((n,m), float)
  
  j = 0
  for i in range(n_samples):
    
    
    # region_counts # num_regions, 2 (free/bound)
    coords, region_counts = dynamics_steps(scene_pixmap, coords, particle_params,
                                           monitor_regions, local_steps, max_occ)
     
    if out_img_dir:
      pixmap = np.array(bg_pixmap)
      add_pixmap_particles(pixmap, bg_pixmap, coords, colors, pixel_decay, pixel_width)
      file_path = os.path.join(out_img_dir, 'step_%05d.png')
      imsave(file_path % i, pixmap)
        
    counts = region_counts.astype(float) / float(local_steps) # average totals per step
    bound = counts[:,0]
    free = counts[:,1]

    count_data.append((bound, free))
                      
    #np.save('npy/particle_coords_%s_%03d.npy' % (label, i), coords)
      
    pixel_points = coords.astype(int)
    hist, ex, ey = np.histogram2d(pixel_points[:,0], pixel_points[:,1], range=((0,n), (0,m)), bins=(n,m))
    density += hist

    j += local_steps
    info("  .. done %d of %d" % (j, n_steps), line_return=True)
  
  count_data = np.array(count_data)
  
  # Normalise and clip extrena
  density /= density.max()
  
  #density = np.clip(density, 0.01, 0.99)
  #density -= density.min()
  #density /= density.max()  
  
  return density, coords, count_data
  


def mem_mol_diff_sim(image_path, regions_path, out_dir, n_samples=DFLT_N_SAMPLES,
                     n_steps=DFLT_N_STEPS, timestep=DFLT_TIMESTEP, d_bound=DFLT_D_BOUND, d_free=DFLT_D_FREE,
                     init_steps=DFLT_INIT_STEPS, k_on=DFLT_K_ON, alt_k_on=DFLT_ALT_K_ON,
                     k_off=DFLT_K_OFF, alt_k_off=DFLT_ALT_K_OFF, max_occ=DFLT_MAX_OCC, write_state_img=False):
  
  from scipy import ndimage  
  from numpy.random import uniform
  
  if os.path.exists(out_dir):
    warn('Output directory "%s" exists. This may overwrite existing files.' % out_dir)
  else:
    os.mkdir(out_dir)
    
  check_invalid_file(image_path)
  check_invalid_file(regions_path)
  
  rois = _read_roi_file(regions_path)
  
  out_count_file = os.path.join(out_dir, 'particle_counts.tsv')
  out_ratio_file = os.path.join(out_dir, 'region_bound_ratios.tsv')
  out_img_path = os.path.join(out_dir, 'particle_density.png')
  
  if write_state_img:
    out_img_dir = os.path.join(out_dir, 'sampled_images')
    
    if not os.path.exists(out_img_dir):
      os.mkdir(out_img_dir)
      
  else:
    out_img_dir = None
      
  try:
    #bg_pixmap = (imread(image_path) * 255).astype(np.uint8)
    bg_pixmap = imread(image_path)
  
  except Exception as err:
    report(str(err), False)
    critical('Failed to load image file.')  
  
  info('Running memodis on membrane image "%s"' % image_path)
  info('Using regions "%s"' % regions_path)
  
  # From origins
  
  origin_mask = (bg_pixmap[:,:,0] > 0) & (bg_pixmap[:,:,1] == 0) & (bg_pixmap[:,:,2] == 0) # Red, no green, no blue
  rows, cols = origin_mask.nonzero()
  
  start_coords = np.array(zip(rows, cols), float)
  n_particles = len(start_coords)
  
  if not n_particles:
    critical('No particle origins (red pixels) founs in input image map.')  
  
  roi_groups = defaultdict(list)
  num_roi_mems = {}
  monitor_regions = []
  for i, (x, y, r, m, g) in enumerate(rois):
    roi_groups[g].append(i)
    num_roi_mems[i] = m
    monitor_regions.append([x,y,r])
  
  monitor_regions = np.array(monitor_regions, float)

  info('Simulating %d particles for %d steps' % (n_particles, n_steps))
  density, coords, count_data = run_dynamics(bg_pixmap, start_coords, out_img_dir,
                                             monitor_regions, n_samples, n_steps, timestep,
                                             d_bound, d_free, init_steps,
                                             k_on, alt_k_on, k_off, alt_k_off, max_occ)
  
  membrane_mask = (bg_pixmap[:,:,0] == 0) & (bg_pixmap[:,:,1] == 0) & (bg_pixmap[:,:,2] == 0)
  #membrane_mask[origin_mask] = False
  
  med = np.median(density[membrane_mask])
  print med
  
  density /= 2.0 * med
  density = np.clip(density, 0.0, 1.0)
  
    
  red = np.interp(density, GRAD_STOPS, GRAD_RED)
  green = np.interp(density, GRAD_STOPS, GRAD_GREEN)
  blue = np.interp(density, GRAD_STOPS, GRAD_BLUE)
  
  red = ndimage.convolve(red, IMG_FILTER)
  green = ndimage.convolve(green, IMG_FILTER)
  blue = ndimage.convolve(blue, IMG_FILTER)  
  
  color_pixmap = np.dstack([red, green, blue])
  color_pixmap -= color_pixmap.min()
  color_pixmap /= color_pixmap.max()
  
  color_pixmap = np.clip(color_pixmap, 0.0, 1.0)
  color_pixmap = (color_pixmap * 255).astype(np.uint8)   
   
  imsave(out_img_path, color_pixmap)
  info('Written %s' % out_img_path)
  
  
  # Write count data
  #  - count_data is for each ROI, (bound, free)
  
  with open(out_count_file, 'w') as out_file_obj:
    write = out_file_obj.write
     
    for j, region in enumerate(monitor_regions):
      m = num_roi_mems[j]
      x_pos, y_pos, radius = monitor_regions[j]
      line = '##region:%d\tx_pos:%d\ty_pos:%d\tradius:%d\tmembranes:%d\n' % (j, x_pos, y_pos, radius, m)
      write('\t'.join(line) + '\n')
       
      head = ['#region', 'step', 'mean_bound', 'mean_free', 'bound/membrane', 'free/membrane']
      write('\t'.join(head) + '\n')
    
      for i, (bound, free) in enumerate(count_data): # for each sample
        line = '%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n'
        write(line % (j, i, bound[j], free[j], bound[j]/m, free[j]/m))
  
  # Write ratios
  # - average over all samples
  mean_count_data = count_data.mean(axis=0)
  
  with open(out_ratio_file, 'w') as out_file_obj:
    write = out_file_obj.write

    head = ['#group_a', 'group_b', 'regions_a', 'regions_b', 'bound_a/membrane', 'bound_b/membrane', 'mean_a', 'mean_b', 'ratio_a/b']
    write('\t'.join(head) + '\n')
    groups = sorted(roi_groups)
    
    if len(groups) > 1:
      bound_dict = defaultdict(list)
      free_dict = defaultdict(list)
    
      for g in groups:
        for j in roi_groups[g]:
          m = num_roi_mems[j]
          bound = mean_count_data[0,j]/m 
          free = mean_count_data[1,j]/m
          bound_dict[g].append(bound)
          free_dict[g].append(free)
    
      for i, a in enumerate(groups[:-1]):
        regions_a = len(bound_dict[a])
        bound_a = sum(bound_dict[a])
        mean_a = np.mean(bound_dict[a])
        
        for j, b in enumerate(groups[i+1:], i+1):
          regions_b = len(bound_dict[b])
          bound_b = sum(bound_dict[b])
          mean_b = np.mean(bound_dict[b])
         
          if mean_b:
            ratio = '%.4f' % (mean_a/mean_b)
          else:
            ratio = ''
          
          data = (a, b, regions_a, regions_b, bound_a, bound_b, mean_a, mean_b, ratio)
          line = '%s\t%s\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%s\n' % data
          write(line)

test_imports()


def main(argv=None):

  from argparse import ArgumentParser
  
  if argv is None:
    argv = sys.argv[1:]
  
  # 'Example use: ./memodis example/cell_boundaries.png example/regions_example.tsv -o example/ '
  
  epilog = 'For further help email tjs23@cam.ac.uk'
  
  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='IMAGE_FILE', dest='in',
                         help='Cell membrane boundary pixmap image in PNG format (other formats may be usable if the PIL library is available).' \
                              ' Non-membrane/cytompasm areas must be perfectly black. Membrane pixels may be any other colour.')
  
  arg_parse.add_argument(metavar='REGIONS_FILE', dest='r',
                         help='A white-space separated text file specifying circular monitored regions of interest. ' \
                              'See the main GitHub README for a description of the file format.')

  arg_parse.add_argument('-o', metavar='OUT_DIR', dest='o', default='.',
                         help='Optional output directory for writing image and count data. Defaults to the current working directory.')
  
  arg_parse.add_argument('-s', '--num_samples', default=DFLT_N_SAMPLES, metavar='NUM_SAMPLES', type=int, dest='s',
                         help='The number of points in the simulation to record particle positions,' \
                              ' after the initialization period. Default: %d' % DFLT_N_SAMPLES)

  arg_parse.add_argument('-n', '--num_steps', default=DFLT_N_STEPS, metavar='NUM_SIM_STEPS', type=int, dest='n',
                         help='Number of discrete simulation diffusion steps after the initialization period. Default: %d' % DFLT_N_STEPS)

  arg_parse.add_argument('-t', '--timestep', default=DFLT_TIMESTEP, metavar='TIMESTEP', type=float, dest='t',
                         help='The time interval corresponding to each simulation step. Default: %.4f' % DFLT_TIMESTEP)
  
  arg_parse.add_argument('-b', '--diff-bound', default=DFLT_D_BOUND, metavar='DIFF_BOUND', type=float, dest='b',
                         help='Diffusion coefficient for membrane bound particles, in units of um^2/s. Default: %.4f' % DFLT_D_BOUND)
  
  arg_parse.add_argument('-f', '--diff-free', default=DFLT_D_FREE, metavar='DIFF_FREE', type=float, dest='f',
                         help='Diffusion coefficient for free, unbound particles, in units of um^2/s. Default: %.4f' % DFLT_D_FREE)

  arg_parse.add_argument('-i', '--num-init-steps', default=DFLT_INIT_STEPS, metavar='NUM_INIT_STEPS', type=int, dest='i',
                         help='Number of initialization simulation steps (to equilibriate the system)' \
                              ' before recording particle postions. Default: %d' % DFLT_INIT_STEPS)

  arg_parse.add_argument('-a', '--k-on', default=DFLT_K_ON, metavar='ASSOC_CONST', type=float, dest='a',
                         help='The general membrane association constant, K_on. Default: %.4f' % DFLT_K_ON)
  
  arg_parse.add_argument('-a2', '--alt-k-on', default=DFLT_ALT_K_ON, metavar='ALT_ASSOC_CONST', type=float, dest='a2',
                         help='The special membrane association constant, K_on, for adjusted (red) membranes. Default: %.4f' % DFLT_ALT_K_ON)
  
  arg_parse.add_argument('-d', '--k-off', default=DFLT_K_OFF, metavar='DISSOC_CONST', type=float, dest='d',
                         help='The general membrane dissociation constant, K_off. Default: %.4f' % DFLT_K_OFF)

  arg_parse.add_argument('-d2', '--alt-k-off', default=DFLT_ALT_K_OFF, metavar='ALT_DISSOC_CONST', type=float, dest='d2',
                         help='The special membrane dissociation constant, K_off, for adjusted (red) membranes. Default: %.4f' % DFLT_ALT_K_OFF)

  arg_parse.add_argument('-m', '--max-occ', default=DFLT_MAX_OCC, metavar='MAX_OCCUPANCY', type=int, dest='m',
                         help='Maximum per-pixel particle occupancy at membrane. Default: %d' % DFLT_MAX_OCC)

  arg_parse.add_argument('-img', '--write-images', default=False, action='store_true', dest="img",
                         help='Write sampled states as separate images. Note this slows the simulation somewhat.')
  
  args = vars(arg_parse.parse_args(argv))

  image_path = args['in']
  regions_path = args['r']
  out_dir = args['o']
  n_samples = args['s']
  n_steps = args['n']
  timestep = args['t']
  d_bound = args['b']
  d_free = args['f']
  init_steps = args['i']
  k_on = args['a']
  alt_k_on = args['a2']
  k_off = args['d']
  alt_k_off = args['d2']
  max_occ = args['m']
  write_state_img = args['img']
                   
  mem_mol_diff_sim(image_path, regions_path, out_dir, n_samples, n_steps, timestep,
                   d_bound, d_free, init_steps, k_on, alt_k_on, k_off, alt_k_off, max_occ, write_state_img)
  
if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()

  
