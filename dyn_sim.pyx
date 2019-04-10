from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt, exp, ceil
from numpy cimport ndarray, uint8_t, int32_t
from numpy import array, zeros, int32
from numpy.random import normal

def dynamics_steps(ndarray[uint8_t, ndim=3] scene,
                   ndarray[double, ndim=2] coords,
                   ndarray[double, ndim=2] particle_params,
                   ndarray[double, ndim=2] zone_params,
                   ndarray[double, ndim=2] regions,
                   py_num_steps=8, py_max_occ=1):
    
  cdef int32_t i, j, k, ka, kb, k0, k1, k2, a, b, a0, a1, b0, b1, sign, s, free
  cdef int32_t px, py, px0, py0
  cdef int32_t n_coords = len(coords)
  cdef int32_t n_steps = py_num_steps
  cdef int32_t max_occ = py_max_occ
  cdef int32_t px_lim, py_lim, z, n_zones = len(zone_params)
  cdef int32_t n_regions = len(regions)
  cdef double x, y, x1, y1, dx, dy, f, g, r, r2, d2, x_lim, y_lim, p_bind, p_unbind, zx0, zy0, sx, sy
  cdef double p_accept, n_edge, n_free, n_edge0, n_free0
  cdef ndarray[double, ndim=2] deltas
  cdef ndarray[int32_t, ndim=2] adjasent8 = array([[-1, -1], [-1, 0], [-1, 1], [ 0, 1],
                                                   [ 1,  1], [ 1, 0], [ 1,-1], [ 0,-1]], int32)
  cdef ndarray[int32_t, ndim=2] adjasent4 = array([[-1, 0], [ 0, 1],
                                                   [ 1, 0], [ 0,-1]], int32)
  cdef ndarray[int32_t, ndim=2] region_counts = zeros((len(regions), 2), int32)
  
  a = scene.shape[0]
  b = scene.shape[1]
  
  cdef ndarray[int32_t, ndim=2] block_mask = zeros([a, b], int32)
  
  px_lim = b-2
  py_lim = a-2
  
  scene = scene.copy()
  
  for i in range(a):
    for j in range(b):
      scene[i,j,0] = scene[i,j,:3].max() # Non-membrane zone
  
  for i in range(a):
    for j in range(b):
      s = 0
      
      for k in range(4):
        ka = (i + adjasent4[k,0]) % a
        kb = (j + adjasent4[k,1]) % b
        
        if scene[ka,kb,0] == 0:
          s += 1
      
      scene[i,j,1] = s # Number of free edges

  for i in range(a):
    for j in range(b):
      s = 0
      
      if scene[i,j,0] > 0 and scene[i,j,1] == 0: # Membrane and no edges
        
        for k in range(4):
          ka = (i + adjasent4[k,0]) % a
          kb = (j + adjasent4[k,1]) % b
        
          if scene[ka,kb,1] > 0: # Neighbour has a free edge
            s += 2 ** (k+1)
        
      scene[i,j,2] = s # Adjascent membrane pixels which themselves have an edge
        
  
  x_lim = <double>px_lim
  y_lim = <double>py_lim
  
  # set initial block mask
  
  for i in range(n_coords):
    py0 = <int32_t>coords[i,0]
    px0 = <int32_t>coords[i,1]
    block_mask[py0,px0] += 1
  
  # Main dynamics cycles
  
  n_edge = 1.0
  n_free = <double>(n_coords)
  
  for j in range(n_steps):
    deltas = normal(0.0, 1.0, (n_coords, 2))
    n_edge0 = 0.0
    n_free0 = 0.0
    
    for i in range(n_coords):
      
      py0 = <int32_t>coords[i,0]
      px0 = <int32_t>coords[i,1]
      
      p_accept = min((p_bind * n_free)/max(0.01, n_edge), 1.0)

      # unset prev block mask
      block_mask[py0,px0] -= 1

      # Set different (un)binding for particles within zone radii
      for z in range(n_zones):
        zx0 = zone_params[z,0] # xPos
        zy0 = zone_params[z,1] # yPos
        r2 = zone_params[z,2] * zone_params[z,2] # radius ^ 2
        
        y = coords[i,0]-zy0
        x = coords[i,1]-zx0
        d2 = x*x + y*y
        
        if d2 < r2:
          p_bind   = particle_params[i,4] # Adjusted (un)binding 
          p_unbind = particle_params[i,5] #
          break
          
      else:
        p_bind = particle_params[i,2] # Basic (un)binding
        p_unbind = particle_params[i,3]
            
      
      #
      
      if p_bind  > 1.0:
        p_bind = 1.0
      
      elif p_bind < 0.0:
        p_bind = 0.0      
      
      if p_unbind  > 1.0:
        p_unbind = 1.0
      
      elif p_unbind < 0.0:
        p_unbind = 0.0
             
      y = coords[i,0]
      x = coords[i,1]
           
      if scene[py0,px0,0] > 0: # Bound
        free = 0
       
        if p_unbind*RAND_MAX > rand(): # Unbind
          k0 = rand() % 4
          
          for k in range(4):
            ka = (k+k0) % 4 
            a = adjasent4[ka,0]
            b = adjasent4[ka,1]
            py = py0 + a
            px = px0 + b
           
            if py < 0:
              continue
            if px < 0:
              continue
            if px > px_lim:
              continue
            if py > py_lim:
              continue      
 
            if scene[py,px,0] < 1:
              break
              
          y += <double>a
          x += <double>b
 
        else:
          
          k0 = rand() % 8
          r = particle_params[i,1] * deltas[i,0] # Sigma bound, Linear diffusion 
          r2 = r * r
          d2 = 0.0
          
          if rand() % 2 == 1:
            sign = -1
            a0 = 7
            b0 = -1
          else:
            sign = 1
            a0 = 0
            b0 = 8
          
          while d2 < r2: # Creep along pixel border ; along centre-centre lines
            
            for k in range(a0, b0, sign):
              kb = (k0 + k) % 8            
              a = adjasent8[kb,0]
              b = adjasent8[kb,1]
              py = py0 + a
              px = px0 + b
              
              if py < 0:
                continue
              if px < 0:
                continue
              if px > px_lim:
                continue
              if py > py_lim:
                continue

              if (scene[py,px,0] > 0) and (scene[py,px,1] > 0): # Is in membrane and has an edge
                k0 = kb
                break
             
            else: # No possibilities ; never hit
              break
            
            # target is centre of adjascent destination pixel
            
            y1 = <double>py + 0.5
            x1 = <double>px + 0.5
            
            # move particle a fraction of destination direction
            
            f = sqrt(<double>(a*a + b*b))
            
            dy = max(r, 1.0)/f * (y1 - y) # Make sure membrane cannot be crossed in a single jump
            dx = max(r, 1.0)/f * (x1 - x)
            
            # can go toward but not cross over destination pixel centre
            
            if (dx < 0.0 and x+dx < x1) or (dx > 0.0 and x+dx > x1):
              dx = x1 - x
              dy = y1 - y            
             
            elif (dy < 0.0 and y+dy < y1) or (dy > 0.0 and y+dy > y1):
              dy = y1 - y            
              dx = x1 - x
            
            x += dx
            y += dy
            
            d2 += (dx*dx + dy*dy) # Limit is square
          
            py0 = <int32_t>y # Often the underlying pixel does not change
            px0 = <int32_t>x
            
            #print d2, r2, dx, dy
            
        if y < 1.0:
          y = 1.0

        if x < 1.0:
          x = 1.0
 
        if y > y_lim:
          y = y_lim
 
        if x > x_lim:
          x = x_lim

        coords[i,0] = y
        coords[i,1] = x
      
      else: # Free
        n_free0 += 1.0
        free = 1
        
        # diffuse
        
        dx = particle_params[i,0] * deltas[i,0]
        dy = particle_params[i,0] * deltas[i,1]
        
        y = coords[i,0]
        x = coords[i,1]
         
        px = px0
        py = py0
        
        # Potential endpoint
        y1 = y + dy
        x1 = x + dx
        
        # The trajectory for the timestep        
        if abs(dy) > abs(dx):
          sy = min(1.0, abs(dy))
          
          if dy < 0.0:
            sy *= -1.0
          
          sx = sy * dx / dy
          
        else:
          sx = min(1.0, abs(dx))
          
          if dx < 0.0:
            sx *= -1.0
          
          sy = sx * dy / dx
        
        a0 = <int32_t>ceil(dy/sy)
        
        for a in range(a0):
          y += sy
          x += sx
          
          if x < 0.0:
            x = x_lim
          elif x > x_lim:
            x = 0.0

          if y < 0.0:
            y = y_lim
          elif y > y_lim:
            y = 0.0
            
          py = <int32_t>y
          px = <int32_t>x
          
          if scene[py,px,0] > 0:
            n_edge0 += 1.0
            
            if scene[py,px,1] == 0: # No free edges:
              b = scene[py,px,2] # Border edges
              k0 = rand() % 4
              
              for k in range(4):
                k1 = (k + k0) % 4
                
                if (2 ** (k1+1)) & b:
                  py += adjasent4[k1,0]
                  px += adjasent4[k1,1]
                  break
              
            if p_accept*RAND_MAX > rand():
              if block_mask[py,px] < max_occ: # Bind : binding is initially to a pixel centre
                coords[i,0] = <double>py + 0.5
                coords[i,1] = <double>px + 0.5         
                
              else:
                coords[i,0] = y - sy
                coords[i,1] = x - sx
              
            else:
              coords[i,0] = y - sy
              coords[i,1] = x - sx
            
            break
            
        else:
          if x < 0.0:
            x = x_lim
          elif x > x_lim:
            x = 0.0

          if y < 0.0:
            y = y_lim
          elif y > y_lim:
            y = 0.0
        
          coords[i,0] = y
          coords[i,1] = x

        px0 = px
        py0 = py
        
      # set blocking mask
      block_mask[py0,px0] += 1

      # Analysis regions of interest for all cycles
      
      for z in range(n_regions):
        zx0 = regions[z,0] # xPos
        zy0 = regions[z,1] # yPos
        r2 = regions[z,2] * regions[z,2] # radius

        y = coords[i,0]-zy0
        x = coords[i,1]-zx0
        d2 = x*x + y*y
 
        if d2 < r2:
          region_counts[z, free] += 1
    
    n_edge = n_edge0
    n_free = n_free0
    
  return coords, region_counts


def add_pixmap_particles(ndarray[uint8_t, ndim=3] pixmap,
                         ndarray[uint8_t, ndim=3] scene,
                         ndarray[double, ndim=2] coords,
                         ndarray[uint8_t, ndim=1] colors,
                         fade=5, width=1):
  
  cdef int32_t i, j, x, y, n_coords = len(coords)
  cdef int n, m, f, g, a, b, w1, w2
  
  n = len(scene)
  m = len(scene[0])
  
  scene = scene.copy()

  if width % 2 == 0:
    w2 = width/2
    w1 = 1-w2
  else:
    w2 = (width-1)/2
    w1 = -w2
  
  w2 += 1  
  
  f = <int>fade
  g = f-1 
  
  if f > 0:
    for i in range(n):
      for j in range(m):
        pixmap[i,j,0] = <uint8_t>((<int>scene[i,j,0] + g*<int>pixmap[i,j,0])/f)
        pixmap[i,j,1] = <uint8_t>((<int>scene[i,j,1] + g*<int>pixmap[i,j,1])/f)
        pixmap[i,j,2] = <uint8_t>((<int>scene[i,j,2] + g*<int>pixmap[i,j,2])/f)
  
  else:
    for i in range(n):
      for j in range(m):
        pixmap[i,j,0] = scene[i,j,0]
        pixmap[i,j,1] = scene[i,j,1]
        pixmap[i,j,2] = scene[i,j,2]
      
  for i in range(n):
    for j in range(m):
      scene[i,j,0] = scene[i,j,:3].max() # Non-membrane zone
  
  for i in range(n_coords):
    x = <int32_t>coords[i,1]
    y = <int32_t>coords[i,0]
    
    # Add BGRA
    
    if scene[y,x,0] == 0:
      for j in range(3):
        for a in range(w1, w2):
          if 0 <= y + a < n:
            for b in range(w1, w2):
              if 0 <= x + b < m:
                pixmap[y+a,x+b,j] = colors[j]
  
    
    else:
      for j in range(3):
        for a in range(w1, w2):
          if 0 <= y + a < n:
            for b in range(w1, w2):
              if 0 <= x + b < m:
                pixmap[y+a,x+b,j] = colors[j+3] 
  
  return pixmap

  
