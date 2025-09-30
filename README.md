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

Con las **8 variables estandarizadas** del *California Housing*  
(*MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude*), OLS sirve como línea base y Ridge/Lasso se contrastan tanto en error fuera de muestra como en la estructura de los coeficientes $\beta$.

En primer lugar, el **desempeño en prueba** fue prácticamente indistinguible entre los tres modelos lineales en el espacio original:
- **OLS / LinearRegression:** $$ \text{MSE}_{\text{test}} \approx 0.5410, \quad R^2_{\text{test}} \approx 0.6122 $$.
- **Ridge (L2):** $\text{MSE}_{\text{test}}\approx \mathbf{0.5413}$ (muy próximo a OLS).
- **Lasso (L1):** $\text{MSE}_{\text{test}}\approx \mathbf{0.5411}$ (también indistinguible de OLS).

Ahora bien, aunque el error no cambia, **la regularización sí altera la geometría de la solución**:
- **Ridge** encoge las magnitudes $|\beta|$ **de forma suave y continua** cuando crece $\lambda$ (no produce ceros), lo que estabiliza los pesos frente a colinealidad moderada y distribuye señal entre predictores relacionados.
- **Lasso** introduce **esparsidad**: a medida que aumenta $\lambda$, varios coeficientes se vuelven exactamente cero, lo que favorece **parsimonia** e **interpretabilidad**, a costa de mayor variabilidad cuando predictores están fuertemente correlacionados.

Al incorporar **polinomios de grado 2** (8→44 predictores: cuadrados + interacciones), el panorama cambia de manera sustantiva.  
- **Ridge (Poly-2)** con $\alpha^\*\approx \mathbf{2.02}$ alcanza $\text{MSE}_{\text{test}}\approx \mathbf{0.489}$, una mejora clara frente al modelo lineal (~0.54).  
- **Lasso (Poly-2)** con $\alpha^\*\approx \mathbf{0.0126}$ deja **≈68 %** de coeficientes en **0**, pero su $\text{MSE}_{\text{test}}\approx \mathbf{0.553}$ empeora frente a OLS/Ridge.

En suma, **la expansión polinómica introduce fuerte colinealidad** entre $\{x, x^2, x\!\times\!z\}$; en ese entorno, **L2** amortigua la varianza y generaliza mejor al repartir peso entre términos correlacionados, mientras **L1** tiende a “elegir” pocos términos dentro de grupos correlacionados y, en este conjunto, termina descartando interacciones útiles. De ahí que **Ridge (Poly-2)** emerja como la alternativa más robusta ($\text{MSE}_{\text{test}}\approx 0.489$).

---

### B. Effect of learning rate on gradient descent

Para minimizar el MSE con **gradient descent** (misma inicialización; presupuesto fijo de iteraciones) se evaluaron dos tasas \(\eta\).

- \(\eta = 10^{-3}\)
  - La curva de costo es monótona pero convergencia lenta; al tope de pasos no alcanza el óptimo práctico.
  - \( \text{MSE}_{\text{test}} \approx 0.571 \, \text{y} \| \beta_{\text{GD}} - \beta_{\text{OLS}} \|_2 \) no es despreciable (soluciones alejadas).

- \(\eta = 10^{-2}\)
  - Convergencia sustancialmente más rápida y estable.
  - \( \text{MSE}_{\text{test}} \approx 0.541 \), virtualmente igual a OLS; además \( \| \beta_{\text{GD}} - \beta_{\text{OLS}} \|_2 \approx 0.0058 \) (parámetros casi idénticos).

De ello se desprende que, en un problema convexo como el MSE lineal, una tasa moderada (\(\eta = 10^{-2}\)) ofrece el mejor **compromiso velocidad–estabilidad**, reproduce la solución cerrada en métrica y parámetros, y evita tanto la ineficiencia de tasas muy pequeñas como el riesgo de oscilaciones de tasas grandes.

---

### C. How k-fold cross-validation influenced the choice of regularization strength

Se utilizó **5-fold CV** con **Pipeline** para ajustar el preprocesamiento **dentro de cada fold** (evitando *data leakage*) y sintonizar $\alpha$ en Ridge/Lasso.

En el **espacio original (8 variables)**, CV selecciona $\alpha^\*$ en rangos distintos para **L2** y **L1**; sin embargo, los tres enfoques entregan $\text{MSE}_{\text{test}}\approx 0.541$. En estas condiciones de “empate” en error, la decisión razonable se guía por el objetivo: **estabilidad** de coeficientes (Ridge) o **esparsidad/interpretabilidad** (Lasso).

En el **espacio polinomial (44 variables)**, CV revela comportamientos divergentes:
- **Ridge (Poly-2):** $\text{MSE}_{\text{test}}\approx \mathbf{0.489}$ con $\alpha^\*\approx 2.02$ (mejor generalización).  
- **Lasso (Poly-2):** $\text{MSE}_{\text{test}}\approx \mathbf{0.553}$ y **≈68 %** de ceros (parsimonia, pero peor desempeño).

Cabe notar una **tensión CV→test**: el **mínimo de $\text{MSE}_{\text{CV}}$** favorece a **Lasso** (≈**0.529**) sobre **Ridge** (≈**0.633**) en el espacio polinomial, mientras que **en prueba** gana **Ridge** con margen. Esta discrepancia es coherente con la **selección de variables** de L1 bajo fuerte colinealidad (validación más ruidosa entre folds) frente a los **caminos de contracción más suaves** de L2, que suelen traducirse en menor varianza y **mejor generalización**.

Por tanto, considerando simultáneamente CV y desempeño en test, **Ridge con polinomios de grado 2** resulta la elección **más robusta** para este conjunto ($\alpha^\*\approx 2.02$, $\text{MSE}_{\text{test}}\approx 0.489$), mientras que **Lasso (Poly-2)** privilegia interpretabilidad a costa de error más alto.

---

## Síntesis y respuestas para la Parte D

Este apartado aplica los conceptos de regresión lineal, gradiente descendente y regularización al dataset de alquiler de bicicletas (`hour.csv`), recopilado en Washington D.C. durante 2011–2012.

### 1. Preprocesamiento de datos
- Fuente: archivo `hour.csv`, con registros horarios de demanda y condiciones climáticas.  
- Transformaciones aplicadas:
  - Conversión de `dteday` a tipo `datetime`.
  - Agregación a nivel diario (731 observaciones):
    - Suma: variables de conteo (`cnt`, `casual`, `registered`).
    - Promedio: variables climáticas (`temp`, `atemp`, `hum`, `windspeed`).
    - Moda: variables categóricas (`season`, `yr`, `mnth`, `weekday`, `holiday`, `workingday`, `weathersit`).  
- Resultado: dataset diario con variables representativas de clima, calendario y demanda.

---

### 2. Regresión OLS (desde cero)
- Implementación manual de la ecuación normal:  
  \[
  \hat{\beta} = (X^\top X)^{-1}X^\top y
  \]
- Resultados en test:
  - MSE ≈ 583,671  
  - R² ≈ 0.833
- Interpretación: la temperatura y el año incrementan la demanda, mientras que humedad, viento y clima adverso la reducen.

---

### 3. Gradiente Descendente (GD)
- Se implementó GD para minimizar el MSE.  
- Se probaron dos tasas de aprendizaje: η = 0.001 y η = 0.01.  
- Ambos convergieron a soluciones cercanas a OLS.  
- Mejor resultado (η=0.001):
  - MSE ≈ 583,554  
  - R² ≈ 0.833

Las curvas de costo muestran convergencia estable sin oscilaciones.

---

### 4. Baseline con `scikit-learn`
- Se entrenó `LinearRegression` con los mismos datos estandarizados.  
- Los coeficientes y métricas coincidieron exactamente con OLS manual.  
- Confirma la correcta implementación de la ecuación normal y la equivalencia entre ambas aproximaciones.

---

### 5. Regularización (Ridge y Lasso)
- Se evaluaron Ridge y Lasso con validación cruzada (10 folds).  
- Mejores hiperparámetros:
  - Ridge: α ≈ 5.96  
  - Lasso: α ≈ 0.05
- Resultados en test:
  - Ridge → MSE ≈ 584,478, R² ≈ 0.832
  - Lasso → MSE ≈ 583,616, R² ≈ 0.833

Ambos alcanzan desempeño prácticamente idéntico a OLS, con ligeras diferencias:
- Ridge (L2): estabiliza coeficientes en presencia de colinealidad.  
- Lasso (L1): introduce esparsidad, eliminando predictores redundantes.  

---

### 6. Conclusión 
- El modelo lineal describe adecuadamente la demanda de bicicletas diarias (R² ≈ 0.83).  
- OLS, GD y `LinearRegression` producen resultados equivalentes.  
- Ridge y Lasso no mejoran sustancialmente el error en este dataset, pero ofrecen ventajas interpretativas:
  - Ridge: mayor estabilidad de coeficientes.  
  - Lasso: selección automática de variables.  

La Parte D nos confirma que la regularización es útil como herramienta de robustez, aunque en este caso no aporta mejoras significativas en el desempeño predictivo frente al modelo lineal base.
