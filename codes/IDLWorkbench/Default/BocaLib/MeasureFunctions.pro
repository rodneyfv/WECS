PRO MeasureFunctions
   ;...
END


;+
; Compute the confusion matrix from a classification results and a given set of test samples.
;
; @returns a numeric (KxK) matrix, where K is the number of classes
; 
; @param IndexImage {in}{required}{type=numeric} image classification index
;
; @param TestROIs {in}{required}{type=pointer} pointer to a struct with informations about samples
;-
FUNCTION CONFUSION_MATRIX, IndexImage, TestROIs

NClass = N_ELEMENTS(TestROIs)
ConfMatrix = INTARR(NClass,NClass)

FOR i = 0, N_ELEMENTS(TestROIs)-1 DO BEGIN
   Aux = *TestROIs[i]
   FOR j = 0, N_ELEMENTS(Aux.RoiLex)-1 DO BEGIN
      Index = IndexImage[Aux.RoiLex[j]]
      ConfMatrix[Index,i]++
   ENDFOR
ENDFOR

Return, ConfMatrix
END


;+
; Compute concordance measures from a given confusion matrix.
;
; @returns a numeric array with: Overall Accuracy, Tau index of concordance, Tau variance, 
;          Kappa index of concordance and Kappa variance.
; 
; @param ConfMatrix {in}{required}{type=numeric} confusion matrix computed from a classification result
;-
FUNCTION CONCORDANCE_MEASURES, ConfMatrix

Total = FLOAT(TOTAL(ConfMatrix))
m = FLOAT(N_ELEMENTS(ConfMatrix[*,0]))
Diag = 0.0 & Marg = 0.0
FOR i = 0, m-1 DO Diag += ConfMatrix[i,i] 
FOR i = 0, m-1 DO Marg += TOTAL(ConfMatrix[i,*])*TOTAL(ConfMatrix[*,i])
OverAccuracy = Diag/Total
Marg = Marg/(Total^(2))
Tau = (OverAccuracy - (1/m))/(1 - (1/m))
VarTau = (1/Total)*((OverAccuracy*(1-OverAccuracy))/(1-(1/m))^(2))
Kappa = (OverAccuracy - Marg)/(1 - Marg)

;Kappa variance...
T3 = 0.0 & T4 = 0.0
FOR i = 0, m-1 DO T3 += ConfMatrix[i,i]*(TOTAL(ConfMatrix[i,*]) + TOTAL(ConfMatrix[*,i]))
FOR i = 0, m-1 DO BEGIN
   FOR j = 0, m-1 DO T4 += ConfMatrix[i,j]*(TOTAL(ConfMatrix[j,*]) + TOTAL(ConfMatrix[*,i]))^2
ENDFOR
T1 = OverAccuracy
T2 = Marg
T3 = T3/(Total^(2))
T4 = T4/(Total^(3))

VarKappa = (1/Total)*((T1*(1-T1)/(1-T1)^(2)) + $
                      2*((1-T1)*(2*T1*T2-T3)/(1-T2)^(3)) + $
                      ((((1-T1)^(2))*(T4-4*T2^(2)))/(1-T2)^(4)))

Return, [OverAccuracy,Tau,VarTau,Kappa,VarKappa]
END



;############################
PRO WRITE_REPORT, Result, Path
   
   GET_LUN, Unit 
   OPENW, Unit, Path   
   
   Measures = Result.MeasuresIndex
   ConfMatrix = Result.ConfusionMatrix
   
   Printf,Unit,'Measures Indices:'
   Printf,Unit,'' 
   Printf,Unit,'Overall Accuracy... ',Measures[0]
   Printf,Unit,'Tau Agreement Coefficient... ',Measures[1]
   Printf,Unit,'Tau Variance... ',Measures[2] 
   Printf,Unit,'Kappa Agreement Coefficient... ',Measures[3]
   Printf,Unit,'Kappa Variance... ',Measures[4]
   Printf,Unit,''
   Printf,Unit,''
   Printf,Unit,'Confusion Matrix (Under construction!!)'
   Printf,Unit,''
   FOR i = 0, N_ELEMENTS(ConfMatrix[*,0])-1 DO Printf, Unit, STRING(ConfMatrix[*,i])
   
   Close, Unit
END