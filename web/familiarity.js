// The familiarity check (src/waste_sorting_classifier/familiarity.py), version 4: minus the
// smallest squared Mahalanobis distance from the photo's features to a training class mean,
// with the training features' shared inverse covariance. The file holds the class means,
// then the inverse covariance, as float32. Computed in float64 like the Python version.
export function mahalanobis(values, classes, dims, features) {
  const precisionAt = classes * dims;
  const d = new Float64Array(dims);
  let best = Infinity;
  for (let k = 0; k < classes; k++) {
    for (let j = 0; j < dims; j++) d[j] = features[j] - values[k * dims + j];
    let total = 0;
    for (let i = 0; i < dims; i++) {
      const row = precisionAt + i * dims;
      let dot = 0;
      for (let j = 0; j < dims; j++) dot += values[row + j] * d[j];
      total += d[i] * dot;
    }
    if (total < best) best = total;
  }
  return -best;
}
