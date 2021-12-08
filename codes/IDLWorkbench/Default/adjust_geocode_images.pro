PRO Adjust_GeoCode_Images

  Path_base = '/home/rogerio/GIT/repo.wecs/WECS/qgis/rasters/2015-12-26.tif'
  Path_images = '/media/rogerio/Dados/temp.Wecs/instant_images/image_m'
  Path_save = '/media/rogerio/Dados/temp.Wecs/instant_images_geocoded/image_m'

  Result = QUERY_TIFF(Path_base, Info, GEOTIFF=geoVar)
  dimRef = info.DIMENSIONS

  avgImg = FLTARR(dimRef[0],dimRef[1])
  count = 0
  for i = 1, 84 do begin
    path_in = Path_images + strtrim(string(i),1) + '.tiff' 
    path_out = Path_save + strtrim(string(i),1) + '.tiff'
  
    img = read_tiff(path_in)
  
    img_out = CONGRID(img, dimRef[0], dimRef[1])
    write_tiff, path_out, geotiff=geoVar, img_out, /float
  
    avgImg += img_out
    count++
  endfor

  write_tiff, path_out+'eam.tif', geotiff=geoVar, (avgImg*(1.0/count)), /float

  print,'fim...'
END