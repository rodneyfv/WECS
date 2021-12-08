PRO ClassificationFunctions
   ;...
END

;+
; Transform a (KxMxN) rule matrix to a classification image. 
;
; @returns (3xMxN) image classification results.
;
; @param RuleImage {in}{required}{type=numeric matrix} matrix with classification rules
; 
; @param PtrROIs {in}{required}{type=pointer} pointer of a struct with informations about
;        the samples used on the classification process
;-
FUNCTION CLASSIF, RuleImage, PtrROIs

ClaImage = INTARR(3,N_ELEMENTS(RuleImage[*,0,0]),N_ELEMENTS(RuleImage[0,*,0]))

FOR i = 0, N_ELEMENTS(RuleImage[*,0,0])-1 DO BEGIN
   FOR j = 0, N_ELEMENTS(RuleImage[0,*,0])-1 DO BEGIN
      ;modificada para casar com o CP7...
      Index = WHERE(RuleImage[i,j,*] EQ MAX(RuleImage[i,j,*]))
      Roi = *PtrROIs[Index[0]]
      ClaImage[*,i,j] = Roi.RoiColor
   ENDFOR
ENDFOR

Return, ClaImage
END

;+
; Transform a classification result (3xMxN), where each element is a 3d-array that define
; the color class, to a index resulr (1xMxN), where each element represent the class index. 
;
; @returns (1xMxN) image classification index.
;
; @param ClaImage {in}{required}{type=numeric matrix} matrix with color classification values 
; 
; @param PtrROIs {in}{required}{type=pointer} pointer of a struct with informations about
;        the samples used on the classification process
;-
FUNCTION CLASSIF_INDEX, ClaImage, PtrROIs

Dim = SIZE(ClaImage,/DIMENSION)
IndexImage = INTARR(Dim[1],Dim[2])

ColorVec = INTARR(3,N_ELEMENTS(PtrROIs))

FOR i = 0, N_ELEMENTS(PtrROIs)-1 DO BEGIN
   AUX = *PtrROIs[i]
   ColorVec[*,i] = AUX.RoiColor
ENDFOR

FOR i = 0, Dim[1]-1 DO BEGIN
   FOR j = 0, Dim[2]-1 DO BEGIN
      Index = -1
      REPEAT BEGIN
         Index++
         Val = NORM(ClaImage[*,i,j] - ColorVec[*,Index])         
      ENDREP UNTIL (Val EQ 0)
      IndexImage[i,j] = Index
   ENDFOR
ENDFOR

Return, IndexImage
END


;+
; Classify a rule image given a threshold produced. Each element is classified as the class 
; with the great rule value, since this value is greater than a threshold.
;
; @returns (2xMxN) image classified
;
; @param RuleImage {in}{required}{type=numeric matrix} matrix with classification rules
; 
; @param PtrROIs {in}{required}{type=pointer} pointer of a struct with informations about
;        the samples used on the classification process
; @param Rate {in}{required}{type=numeric} is a threshold (level of confiance) of classification
;-
FUNCTION CLASSIF_RATE, RuleImage, PtrROIs, Rate

ClaImage = INTARR(3,N_ELEMENTS(RuleImage[0,*,0]),N_ELEMENTS(RuleImage[0,0,*]))

FOR i = 0, N_ELEMENTS(RuleImage[0,*,0])-1 DO BEGIN
   FOR j = 0, N_ELEMENTS(RuleImage[0,0,*])-1 DO BEGIN
      IF(MAX(RuleImage[*,i,j]) LT Rate) THEN ClaImage[*,i,j] = [0,0,0] $
      ELSE BEGIN 
         Index = WHERE(RuleImage[*,i,j] EQ MAX(RuleImage[*,i,j]))
         Roi = *PtrROIs[Index[0]]
         ClaImage[*,i,j] = Roi.RoiColor
      ENDELSE
   ENDFOR
ENDFOR

Return, ClaImage
END



FUNCTION UNSUPERVISED_FUZZY_CLASSIFICATION, RuleImage, Rate

ClaImage = INTARR(3,N_ELEMENTS(RuleImage[0,*,0]),N_ELEMENTS(RuleImage[0,0,*]))
FOR i = 0, N_ELEMENTS(RuleImage[0,*,0])-1 DO BEGIN
   FOR j = 0, N_ELEMENTS(RuleImage[0,0,*])-1 DO BEGIN
      IF(MAX(RuleImage[*,i,j]) LT Rate) THEN ClaImage[*,i,j] = [0,0,0] $
      ELSE BEGIN 
         Index = WHERE(RuleImage[*,i,j] EQ MAX(RuleImage[*,i,j]))
         if index eq -1 then Index = 0 ;cachorragem!!!
         ClaImage[*,i,j] = TEKTRONIX(Index)
      ENDELSE
   ENDFOR
ENDFOR

Return, ClaImage
END



FUNCTION TEKTRONIX, Index
;R = [0,100,100,0,0,0,100,100,100,60,0,0,55,100,33,67,100,75,45,17,25,50,75,100,67,40,17,17,17,45,75,90]
;G = [0,100,0,100,0,100,0,100,50,83,100,50,0,0,33,67,100,100,100,100,83,67,55,33,90,90,90,67,50,33,17,9]
;B = [0,100,0,0,100,100,83,0,0,0,60,100,83,55,33,67,33,45,60,75,83,83,83,90,45,55,67,90,100,100,100,100]

R = [100,0,0,0,100,100,100,60,0,0,55,100,33,67,100,75,45,17,25,50,75,100,67,40,17,17,17,45,75,90]
G = [0,100,0,100,0,100,50,83,100,50,0,0,33,67,100,100,100,100,83,67,55,33,90,90,90,67,50,33,17,9]
B = [0,0,100,100,83,0,0,0,60,100,83,55,33,67,33,45,60,75,83,83,83,90,45,55,67,90,100,100,100,100]

Return, [(R[Index]*255)/100, (G[Index]*255)/100, (B[Index]*255)/100]
END