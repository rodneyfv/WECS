PRO ImageFunctions
   ;...
END

;+
; Resolve future problems when line/column images are used, adding one more 
; dimension to represent a single band image. If the input image is represented 
; by a line/column matrix, the output is a image with single-band/line/column dimensions 
;
; @returns (kxMxN) matrix, k>=1
;
; @param Image {in}{required}{type=numeric matrix} image matrix representation. 
;-
FUNCTION IMAGE_INSPECT, Image

Dims = size(Image,/dimension)
IF (N_ELEMENTS(Dims) LE 2) THEN BEGIN
   Temp = FLTARR(1,Dims[0],Dims[1]) 
   Temp[0,*,*] = Image
   Return, Temp
ENDIF ELSE Return, Image


END

;####################################
FUNCTION IMAGE_3B, Image

Dims = size(Image,/dimension)
NB = Dims[0]

IF (NB EQ 3) THEN Return, Image

Temp = FLTARR(3,Dims[1],Dims[2])
IF (NB GT 3) THEN FOR i = 0, 2 DO Temp[i,*,*] = Image[i,*,*]   
IF (NB LT 3) THEN FOR i = 0, (NB-1) DO Temp[i,*,*] = Image[i,*,*]
Return, Temp
END

;####################################
FUNCTION ATTRIBUTE_VECTOR, PATH_IMG

Image = READ_TIFF(PATH_IMG)
Image = IMAGE_INSPECT(Image)
Dims = SIZE(Image, /DIMENSION)

Return, INDGEN(Dims[0])
END






;Adicionados em 14-Fev-2012

;#########################
FUNCTION GET_POSITIONS, Lex, NC, NL

;Reading values on Lex and building a vector of positions
Pos = LONARR(2,N_ELEMENTS(Lex))
FOR i=0, N_ELEMENTS(Lex)-1 DO BEGIN
   Pos[0,i] = LONG(Lex[i] MOD NC)
   Pos[1,i] = LONG(Lex[i]/NC)
ENDFOR

Return, Pos
END

;##########################
FUNCTION GET_DIMENSIONS, Image

;Dimensao dos dados
Dim = SIZE(Image, /DIMENSION)
IF N_ELEMENTS(Dim) EQ 2 THEN BEGIN
   NC = Dim[0]
   NL = Dim[1]
ENDIF ELSE BEGIN
   NC = Dim[1]
   NL = Dim[2]
ENDELSE

Return, [NC,NL]
END




;CABRITO!!!!!!!!!!!! LUGAR ERRADO!
;##########################################

FUNCTION GET_NEIGHBORS,pi,pj,WinX,WinY,NC,NL ;O proprio ponto é incluso com contexto! (CORRETO)

Neighs = [0,0]
FOR i = FIX(-WinX)/2, FIX(+WinX)/2 DO BEGIN
   FOR j = FIX(-WinY)/2, FIX(+WinY)/2 DO BEGIN
      IF ((((pi + i) GE 0) AND ((pi + i) LT NC)) AND $
         (((pj + j) GE 0) AND ((pj + j) LT NL)) ) THEN Neighs = [[Neighs],[pi+i,pj+j]]    
   ENDFOR
ENDFOR

Return, Neighs[*,1:N_ELEMENTS(Neighs[0,*])-1]
END