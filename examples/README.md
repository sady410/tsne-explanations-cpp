# Iris example

`iris.csv` contains 150 samples, four numeric features, and a `Targets` class
column. The Python workflow detects its semicolon delimiter automatically.

```sh
python tools/run_and_visualize.py \
  --input examples/iris.csv \
  --label-column Targets \
  --standardize \
  --perplexity 30 \
  --iterations 400 \
  --binary-dir build \
  --output-dir results/iris \
  --sample 100
```

On a Ninja build stored in `build/clang`, use `--binary-dir build/clang`.
