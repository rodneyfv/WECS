PRO ProbabilityFunctions
   ;...
END

;+
; Compute a probability of a given vector from a Gaussian Multivariated function 
; with parameters Mu (mean vector) and Sigm (covariance matrix).  
;
; @returns a probability value.
; 
; @param X {in}{required}{type=numeric} n-dimensional real vector
; 
; @param Mu {in}{required}{type=numeric} n-dimensional real vector
; 
; @param Sigm {in}{required}{type=numeric} n-square real matrix
;-
FUNCTION MULTIV_GAUSS, X, Mu, Sigm

Dim = SIZE(X,/DIMENSION)

Value = (1/((2*!PI)^(Dim/2.0) * (DETERM(Sigm, /double))^(0.5))) *  $
        EXP( -0.5d * (X - Mu) ## INVERT(Sigm, /double) ## TRANSPOSE(X - Mu) )
        
Return, Value
END

;##########################################
FUNCTION MULTIV_GAUSS_CUSTOM, X, Mu, Sigm

Dim = SIZE(X,/DIMENSION)

A = (1/((2*!PI)^(Dim/2.0) * (DETERM(Sigm, /double))^(0.5)))
B = EXP( -0.5 * (X - Mu) ## INVERT(Sigm, /double) ## TRANSPOSE(X - Mu) )
        
IF FINITE(A/B) THEN Return, A/B ELSE Return, 0
END