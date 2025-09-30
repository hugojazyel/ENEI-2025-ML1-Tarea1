# Tarea 1: Modelos Lineales, Regularización y Selección de Modelos con Datos Reales

**Deadline:** Lunes, 29 de septiembre de 2025, 23:59.

**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

---
## **Integrantes de grupo**
- Calle Ontaneda, Hugo Jazyel
- Chero Villegas, Leidy Fabiola
- Cueva Mendoza, Jherson Aldair.

---

## Síntesis y respuestas para la Parte A, B y C

### A. Differences observed between OLS, Ridge, and Lasso

Con las 8 variables estandarizadas del conjunto de datos *California Housing* (*MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude*), OLS sirve como línea base y Ridge/Lasso se contrastan tanto en error fuera de muestra como en la estructura de los coeficientes $\beta$.

En primer lugar, el desempeño en prueba fue prácticamente indistinguible entre los tres modelos lineales en el espacio original:
- **OLS / LinearRegression:** $\text{MSE}_{\text{test}}\approx \mathbf{0.5410}$, R^2_test ≈ 0.6122.
- **Ridge (L2):** $\text{MSE}_{\text{test}}\approx \mathbf{0.5413}$ (extremadamente próximo a OLS).
- **Lasso (L1):** $\text{MSE}_{\text{test}}\approx \mathbf{0.5411}$ (también indistinguible de OLS).

Ahora bien, aunque el error no cambia, la regularización sí altera la geometría de la solución:
- **Ridge** encoge las magnitudes $|\beta|$ de forma suave y continua cuando crece $\lambda$ (no produce ceros), lo que estabiliza los pesos frente a colinealidad moderada y distribuye señal entre predictores relacionados.
- **Lasso** introduce esparsidad: a medida que aumenta $\lambda$, varios coeficientes se vuelven exactamente cero, lo que favorece parsimonia e interpretabilidad, a costa de mayor variabilidad cuando predictores están fuertemente correlacionados.

Al incorporar polinomios de grado 2 (8→44 predictores: cuadrados + interacciones), el panorama cambia de manera sustantiva.  
- **Ridge (Poly-2)** con $\alpha^\*\approx \mathbf{2.02}$ alcanza $\text{MSE}_{\text{test}}\approx \mathbf{0.489}$, una mejora clara frente al modelo lineal (~0.54).  
- **Lasso (Poly-2)** con $\alpha^\*\approx \mathbf{0.0126}$ deja ≈68% de coeficientes en 0, pero su $\text{MSE}_{\text{test}}\approx \mathbf{0.553}$ empeora frente a OLS/Ridge.

En suma, la expansión polinómica introduce fuerte colinealidad entre $\{x, x^2, x\!\times\!z\}$; en ese entorno, L2 amortigua la varianza y generaliza mejor al repartir peso entre términos correlacionados, mientras L1 tiende a “elegir” pocos términos dentro de grupos correlacionados y, en este conjunto, termina descartando interacciones útiles. Es así como Ridge (Poly-2) emerge como la alternativa más robusta ($\text{MSE}_{\text{test}}\approx 0.489$).

---

### B. Effect of learning rate on gradient descent

Para minimizar el MSE con gradient descent (misma inicialización y presupuesto fijo de iteraciones) se evaluaron dos tasas de aprendizaje $\eta$.

- **$\eta=10^{-3}$** (0.001)
  - La curva de costo es monótona pero presenta convergencia lenta, en tanto que al tope de pasos no alcanza el óptimo práctico.  
  - $\text{MSE}_{\text{test}}\approx \mathbf{0.571}$ y la distancia entre parámetros GD y OLS (||β_GD − β_OLS||_2) no es despreciable (soluciones alejadas en ≈ 0.609).

- **$\eta=10^{-2}$** (0.01)
  - Esta tasa de aprendizaje, por el contrario, presenta una convergencia sustancialmente más rápida y estable.  
  - $\text{MSE}_{\text{test}}\approx \mathbf{0.541}$ básicamente igual a OLS; además, ||β_GD − β_OLS||_2 ≈ 0.0058 (parámetros casi idénticos).

De ello se desprende que, en un problema convexo como el MSE lineal, una tasa moderada ($\eta=10^{-2}$) ofrece el mejor compromiso velocidad–estabilidad, reproduce la solución cerrada en métrica y parámetros, y evita tanto la ineficiencia de tasas muy pequeñas como el riesgo de oscilaciones de tasas grandes, por lo que se consolida como la mejor tasa entre las dos utilizadas en el experimento.

---

### C. How k-fold cross-validation influenced the choice of regularization strength

En este caso, se utilizó 5-fold CV con Pipeline para ajustar el preprocesamiento dentro de cada fold (evitando *data leakage*) y sintonizar $\alpha$ en Ridge/Lasso.

En el espacio original (8 variables), CV selecciona $\alpha^\*$ en rangos distintos para L2 y L1; sin embargo, los tres enfoques entregan $\text{MSE}_{\text{test}}\approx 0.541$. En estas condiciones de “empate” en error, la decisión razonable se guía por el objetivo: estabilidad de coeficientes (Ridge) o esparsidad/interpretabilidad (Lasso).

En el espacio polinomial (44 variables), CV revela comportamientos divergentes:
- **Ridge (Poly-2):** $\text{MSE}_{\text{test}}\approx \mathbf{0.489}$ con $\alpha^\*\approx 2.02$ (mejor generalización).  
- **Lasso (Poly-2):** $\text{MSE}_{\text{test}}\approx \mathbf{0.553}$ y ≈68% de ceros (parsimonia, pero peor desempeño).

Cabe notar una tensión CV→test: el mínimo de $\text{MSE}_{\text{CV}}$ favorece a Lasso (≈0.529) sobre Ridge (≈0.633) en el espacio polinomial, mientras que en prueba gana Ridge con margen. Esta discrepancia es coherente con la selección de variables de L1 bajo fuerte colinealidad (validación más ruidosa entre folds) frente a los caminos de contracción más suaves de L2, que suelen traducirse en menor varianza y mejor generalización.

Por tanto, considerando simultáneamente CV y desempeño en test, Ridge con polinomios de grado 2 resulta la elección más robusta para este conjunto ($\alpha^\*\approx 2.02$, $\text{MSE}_{\text{test}}\approx 0.489$), mientras que Lasso (Poly-2) privilegia interpretabilidad a costa de error más alto.

---

## Síntesis y respuestas para la Parte D

Este apartado aplica **regresión lineal**, **descenso por gradiente** y **regularización** a la **demanda diaria** construida desde `hour.csv` (Washington D.C., 2011–2012).

### A. Differences observed between OLS, Ridge, and Lasso

Usando agregados diarios (731 días) con predictores estandarizados de clima (promedios de `temp`, `atemp`, `hum`, `windspeed`) y calendario (`season`, `yr`, `mnth`, `weekday`, `holiday`, `workingday`, `weathersit`):

- **Desempeño en test**
  - **OLS (ecuación normal)**: **MSE ≈ 583 671**, **R² ≈ 0.833**.  
  - **Ridge (L2)** con **α\* ≈ 5.96**: **MSE ≈ 584 478**, **R² ≈ 0.832**.  
  - **Lasso (L1)** con **α\* ≈ 0.05**: **MSE ≈ 583 616**, **R² ≈ 0.833**.

- **Estructura de coeficientes**
  - **Ridge** encoge magnitudes **suavemente y de forma continua** al aumentar α; estabiliza pesos ante correlaciones (p. ej., `temp` vs. `atemp`).  
  - **Lasso** induce **esparsidad**, apagando predictores débiles o redundantes y facilitando la interpretación **sin** penalizar el error aquí.

- **Signos esperados**
  - `temp` y `yr` se asocian positivamente con la demanda diaria; `hum`, `windspeed` y estados climáticos adversos (`weathersit`) la reducen.

En conjunto, no hay ganancia sustantiva de MSE frente a OLS; la regularización aporta beneficios **estructurales** (estabilidad con L2, parsimonia con L1) más que mejoras de precisión en este conjunto.

---

### B. Effect of learning rate on gradient descent

Se implementó GD para minimizar el MSE del modelo lineal; se evaluaron dos tasas:

- **η = 0.001**  
  - Convergencia **estable** (más lenta).  
  - **MSE(test) ≈ 583 554**, **R² ≈ 0.833** (indistinguible de OLS).

- **η = 0.01**  
  - Convergencia **más rápida** y sin oscilaciones en este problema convexo.  
  - Métricas **prácticamente idénticas** a OLS.

Dado que el MSE lineal es convexo y queda bien condicionado tras estandarizar, **GD converge a la solución de OLS** con tasas moderadas; η muy pequeña ⇒ ineficiencia, η muy grande ⇒ riesgo de oscilación/divergencia. En el rango probado, ambas tasas son válidas y reproducen la solución cerrada.

---

### C. How k-fold cross-validation influenced the choice of regularization strength

Se usó validación cruzada K=10 para sintonizar α en Ridge y Lasso dentro de un `Pipeline` (el *scaler* se ajusta en cada fold, evitando *data leakage* y replicando la transformación en test).

- **α\* seleccionados**
  - **Ridge:** α\* ≈ **5.96**.  
  - **Lasso:** α\* ≈ **0.05**.

- **Efecto en el desempeño**
  - Con esos α\*, **Ridge** y **Lasso** quedan a la par de OLS (R² ≈ 0.83), lo que sugiere buena relación señal–ruido y ausencia de sobreajuste grave en el espacio lineal diario.  
  - Aunque el MSE no mejora, la CV ancla α en zonas de control de varianza (Ridge) o parsimonia (Lasso) preservando la generalización.

En síntesis, para Bike Rentals el modelo lineal ya captura la mayor parte de la variación** (R² ≈ 0.83). La regularización sintonizada por k-fold CV mejora sobre todo la robustez (L2) o la interpretabilidad (L1) sin sacrificar precisión, guiando α hacia soluciones estables y parsimoniosas.
