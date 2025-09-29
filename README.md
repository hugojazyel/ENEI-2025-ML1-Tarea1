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

Con las **8 variables originales estandarizadas**:

- **Desempeño en prueba.**  
  - **OLS / LinearRegression**: MSE(test) ≈ **0.5410**, R²(test) ≈ **0.6122** (coinciden en parámetros y métricas).  
  - **Ridge (L2)**: MSE(test) ≈ **0.5413** (muy próximo a OLS).  
  - **Lasso (L1)**: MSE(test) ≈ **0.5411** (también indistinguible de OLS).
- **Estructura de coeficientes.**  
  - **Ridge** contrae magnitudes de forma **suave y continua** al crecer λ; **no** fuerza ceros, lo que estabiliza pesos ante correlaciones.  
  - **Lasso** **induce esparsidad**: varios coeficientes se anulan exactamente para λ moderados–altos, favoreciendo interpretación y selección de variables.
- **Lectura técnica.** En este espacio relativamente pequeño, la regularización **no reduce** el error de prueba respecto a OLS; su valor se observa en **estabilidad** (Ridge) e **interpretabilidad** (Lasso).

Con **polinomios grado 2** (de 8 a **44** predictores, cuadrados + interacciones):

- **Ridge (Poly‑2)** con α* ≈ **2.02** alcanza **MSE(test) ≈ 0.489**, mejorando con holgura frente al modelo lineal (~0.54).  
- **Lasso (Poly‑2)** con α* ≈ **0.0126** deja ~**68%** de coeficientes **exactamente en 0**, pero su **MSE(test) ≈ 0.553** empeora frente a OLS/Ridge.  
- **Interpretación.** La expansión polinómica introduce fuerte colinealidad; **L2** reduce varianza y reparte señal entre términos relacionados (mejor generalización), mientras **L1** puede descartar interacciones útiles al “elegir” pocos términos dentro de grupos correlacionados (pérdida de desempeño).

---

### B. Effect of learning rate on gradient descent

Se implementó GD sobre el MSE y se evaluaron **dos tasas**:

- **η = 1e‑3 (0.001):**  
  - Curva de costo **monótona decreciente**, pero **convergencia lenta**.  
  - Queda con **MSE(test) ≈ 0.571** al tope de iteraciones (no alcanza por completo la solución de OLS en el presupuesto de pasos).
- **η = 1e‑2 (0.01):**  
  - Convergencia **sustancialmente más rápida** y estable en el costo.  
  - **MSE(test) ≈ 0.541**, prácticamente igual a OLS; además, la **‖β_GD − β_OLS‖₂** es sensiblemente menor (parámetros casi idénticos).

**Conclusión práctica.** En un problema convexo como MSE lineal:  
- η **demasiado pequeña** asegura estabilidad pero **es ineficiente**;  
- η **moderada** logra el **mejor compromiso** velocidad–estabilidad;  
- η **grande** (no usada aquí) suele generar **oscilaciones/divergencia**.  
Dentro del rango probado, **η = 1e‑2** es la elección eficaz.

---

### C. How k‑fold cross‑validation influenced the choice of regularization strength

Se utilizó **5‑fold CV** con **`Pipeline`** (el preprocesamiento se ajusta **dentro de cada fold**, evitando *data leakage*) para seleccionar **α** en Ridge/Lasso.

- **Espacio original (8 variables).**  
  - CV elige α* en rangos **distintos** para L2 y L1, pero ambos producen **MSE(test) ≈ 0.541**, sin mejoras sustantivas sobre OLS.  
  - La decisión se guía por el objetivo: **estabilidad de coeficientes → Ridge**; **esparsidad/interpretabilidad → Lasso**.
- **Espacio polinomial (44 variables).**  
  - CV identifica **α\*** distintos y revela comportamientos divergentes:  
    - **Ridge (Poly‑2)**: **MSE(test) ≈ 0.489** (mejor generalización).  
    - **Lasso (Poly‑2)**: **MSE(test) ≈ 0.553** y **68%** de coeficientes en cero.  
  - **Tensión CV–test.** Aunque el **mínimo de MSE(CV)** favorece a Lasso frente a Ridge en el espacio polinomial, **en test** domina **Ridge**. Esto es coherente con que L1, al hacer selección de variables, presenta una superficie de validación **más ruidosa** ante colinealidad, mientras L2 muestra caminos **más suaves** y termina generalizando mejor.

Por esto último, y con base en todo lo expuesto previamente, a partir de los datos originales, CV confirma que **no hay ganancia de error** y la elección depende de estabilidad vs. esparsidad; con polinomios, CV y el test juntos respaldan **Ridge (Poly‑2)** como la alternativa **más robusta**.


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
