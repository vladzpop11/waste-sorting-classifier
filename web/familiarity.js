// The familiarity check (src/waste_sorting_classifier/familiarity.py): cosine similarity
// of a photo's features to the nearest training photo. The bank holds every training
// photo's features as a unit vector rounded to 8 bits (value * 127). Computed in float64
// in the same way as the Python version.
export function similarity(bank, rows, dims, scale, features) {
  let norm = 0;
  for (let j = 0; j < dims; j++) norm += features[j] * features[j];
  norm = Math.sqrt(norm);
  let best = -Infinity;
  for (let i = 0, o = 0; i < rows; i++, o += dims) {
    let dot = 0;
    for (let j = 0; j < dims; j++) dot += bank[o + j] * features[j];
    if (dot > best) best = dot;
  }
  return best / norm / scale;
}
