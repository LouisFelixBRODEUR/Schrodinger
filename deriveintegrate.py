import numpy as np


IntegralOperator = np.array( [[1, 0, 0, 0, 0, 0],[1, 1, 0, 0, 0, 0],[1,1,1,0, 0, 0],[1,1,1,1,0,0],[1,1,1,1,1,0], [1,1,1,1,1,1]])
print(f"""
	Matrice intégrale trapeze: on suppose que l'element qui precede
	f[0] est zero, alors les trapezes sont obtenus avec cette matrice:

{IntegralOperator}
	"""
	)

DiffOperator = np.linalg.inv(IntegralOperator) 
print(f""" 
	On calcule la matrice
	inverse de cette matrice, (i.e. l'intégrale trapeze). Remarquez que cela
	suppose que l'élément qui precede f[0] est zero.

{DiffOperator}
	"""
)

DFinDiffOperator = np.array( [[-2, 2, 0, 0, 0, 0],[-1,0, 1, 0, 0, 0],[0, -1,0, 1, 0, 0],[0,0,-1,0, 1, 0],[0,0,0,-1,0, 1], [0,0,0,0,-2, 2]])
print(f"""
	Remarquez qu'en calcul différences-finies, on prend la matrice suivante 
	pour la dérivée:

{DFinDiffOperator}
	""")

try:
	print(f"""
	Et si on tente de calculer l'inverse, on obtient que la matrice est singuliere:

	IntegralOperatorFromDFinDiff = np.linalg.inv(DFinDiffOperator)
	""")

	IntegralOperatorFromDFinDiff = np.linalg.inv(DFinDiffOperator)
except Exception as err:
	print(f"	Python Error: {err}")

