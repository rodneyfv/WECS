PRO ADJUSTSIZE


  Path_base = '/home/rogerio/GIT/repo.wecs/WECS/qgis/rasters/2015-12-26.tif'
  Path_in = '/home/rogerio/GIT/repo.wecs/WECS/qgis/rasters/results_copyFigs/forest_changes_time_m30.tiff'
  Path_out = '/home/rogerio/GIT/repo.wecs/WECS/qgis/rasters/results_copyFigs/forest_changes_time_m30__geoCoded.tiff'
  ;--------------------------------------------------------------------------

  imgBase = read_tiff(Path_base)
  imgIn = read_tiff(Path_in)
  
  dims = SIZE(imgBase, /dimensions)
  nb = dims[0]
  nc = dims[1]
  nl = dims[2]

  ;Ajuste de dimensão...
  Result = QUERY_TIFF(Path_base, Info, GEOTIFF=geoVar)
  dimRef = info.DIMENSIONS
  imgOut = CONGRID(imgIn[0,*,*], 1, nc, nl)
  
  ;Salvando o resultado
  write_tiff, path_out, geotiff=geoVar, imgOut

END
