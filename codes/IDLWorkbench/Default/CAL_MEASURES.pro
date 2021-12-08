@ascii_read_roi.pro
@MeasureFunctions.pro

PRO CAL_MEASURES

  path_geral = '/home/rogerio/GIT/repo.wecs/WECS/figs/tiff_images/'
  path_roi = '/home/rogerio/GIT/repo.wecs/WECS/qgis/envi_referenceROIs/Rois_06dez21.txt'
  
  names = IMG_NAMES()
  PtrROIs = ASCII_READ_ROI(path_roi)
  roiChange = *PtrROIs[0]
  roiNonChange = *PtrROIs[1]
  
  Print, 'name;Ac;Pr;Re;F1;MCC;OA;kappa;varKappa;%TP;%TN;%FP;%FN'
  for i = 0, n_elements(names)-1 do begin
    name = names[i]
    path = path_geral+name
    img = read_tiff(path)
    
    ;Determinar/calcular a matriz de confusão...
    mat = LONARR(2,2)
    
    ;Locais de roiChange que são iguais a 1 (TP)
    posTP = where(img[roiChange.RoiLex] eq 1)
    mat[0,0] = N_ELEMENTS(posTP)
    
    ;Locais de roiChange que são iguais a 0 (FN)
    posFN = where(img[roiChange.RoiLex] eq 0)
    mat[0,1] = N_ELEMENTS(posFN)
    
    ;Locais de roiNonChange que são iguais a 1 (FP)
    posFP = where(img[roiNonChange.RoiLex] eq 1)
    mat[1,0] = N_ELEMENTS(posFP)
    
    ;Locais de roiNonChange que são iguais a 0 (TN)
    posTN = where(img[roiNonChange.RoiLex] eq 0)
    mat[1,1] = N_ELEMENTS(posTN)    
    
    measures = CONCORDANCE_MEASURES(mat)

    TP = mat[0,0]+1
    FN = mat[0,1]+1
    FP = mat[1,0]+1
    TN = mat[1,1]+1
    percTP = TP/float(n_elements(roiChange.RoiLex)) * 0.5
    percFP = FP/float(n_elements(roiNonChange.RoiLex)) * 0.5
    percFN = FN/float(n_elements(roiChange.RoiLex)) * 0.5
    percTN = TN/float(n_elements(roiNonChange.RoiLex)) * 0.5
;    percTP = TP/float(n_elements(roiChange.RoiLex)) * 0.5
;    percFP = FP/float(n_elements(roiChange.RoiLex)) * 0.5
;    percFN = FN/float(n_elements(roiNonChange.RoiLex)) * 0.5
;    percTN = TN/float(n_elements(roiNonChange.RoiLex)) * 0.5
    
    Ac = (FLOAT(TP) + TN)/FLOAT(n_elements(roiChange.RoiLex) + n_elements(roiNonChange.RoiLex))
    Pr = FLOAT(TP)/(TP + FP)
    Re = FLOAT(TP)/(TP + FN)
    F1 = (2*Pr * Re)/(Pr + Re)
    MCC = ( FLOAT(TP)*TN - FLOAT(FP)*FN )/SQRT( (FLOAT(TP)+FP)*(FLOAT(FN)+TN)*(FLOAT(FP)+TN)*(FLOAT(TP)+FN))

    line = name+';'+STRTRIM(STRING(Ac),1)+';'+STRTRIM(STRING(Pr),1)+';'+STRTRIM(STRING(Re),1)+';'+STRTRIM(STRING(F1),1)+';'+STRTRIM(STRING(MCC),1)+';'+STRTRIM(STRING(measures[0]),1)+';'+STRTRIM(STRING(measures[3]),1)+';'+STRTRIM(STRING(measures[4]),1)+';'
    line += STRTRIM(STRING(percTP),1)+';'+STRTRIM(STRING(percTN),1)+';'+STRTRIM(STRING(percFP),1)+';'+STRTRIM(STRING(percFN),1)
    Print, line 
  endfor

END


;
;;Locais de roiChange que são iguais a 1 (TP)
;posTP = where(img[roiChange.RoiLex] eq 1)
;mat[0,0] = N_ELEMENTS(posTP)
;
;;Locais de roiChange que são iguais a 0 (FN)
;posFP = where(img[roiChange.RoiLex] eq 0)
;mat[1,0] = N_ELEMENTS(posFP)
;
;;Locais de roiNonChange que são iguais a 1 (FP)
;posFN = where(img[roiNonChange.RoiLex] eq 1)
;mat[0,1] = N_ELEMENTS(posFN)
;
;;Locais de roiNonChange que são iguais a 0 (TN)
;posTN = where(img[roiNonChange.RoiLex] eq 0)
;mat[1,1] = N_ELEMENTS(posTN)



;----------------------------------------
FUNCTION IMG_NAMES
  lista = ['forest_ecs_change_space_KI.tiff',$
    'forest_ecs_change_space_otsu.tiff',$
    'forest_taad_change_space_KI.tiff',$
    'forest_taad_change_space_otsu.tiff',$
    ;'forest_wecs_abscorr_screethrs.tiff',$
    'forest_wecs_change_space_KI.tiff',$
    'forest_wecs_change_space_otsu.tiff' ]
Return, lista
END