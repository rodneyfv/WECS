PRO GEOCODE_RESULTS

  path_geral = '/home/rogerio/GIT/repo.wecs/WECS/figs/tiff_images/'
  Path_base = '/home/rogerio/GIT/repo.wecs/WECS/qgis/rasters/2015-12-26.tif'


  Result = QUERY_TIFF(Path_base, Info, GEOTIFF=geoVar)
  dimRef = info.DIMENSIONS

  names = IMG_NAMES()
 
  for i = 0, n_elements(names)-1 do begin
    name = names[i]
    path = path_geral+name
    img = read_tiff(path)
    
    path_out = path_geral + 'geocoded_' + name

    img_out = CONGRID(img, dimRef[0], dimRef[1])
    write_tiff, path_out, geotiff=geoVar, img_out ;, /float 
  endfor

END




;----------------------------------------
FUNCTION IMG_NAMES
  lista = [$
    'forest_ecs_change_space_KI.tiff',$
    'forest_ecs_change_space_otsu.tiff',$
    'forest_taad_change_space_KI.tiff',$
    'forest_taad_change_space_otsu.tiff',$
    'forest_wecs_abscorr_screethrs.tiff',$
    'forest_wecs_change_space_KI.tiff',$
    'forest_wecs_change_space_otsu.tiff'] 
;    'forest_ecs.tiff',$
;    'forest_taad.tiff',$
;    'forest_wecs_abscorr.tiff']
  Return, lista
END