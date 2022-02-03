Our currently defined recongition threshold [of 0.39](./config.py#L51-L52) is derived through the [`lfw_eval.py`](./lfw_eval.py) command line script.

It groups the images of [the LFW data set](http://vis-www.cs.umass.edu/lfw/#resources) in [R & L pairs](./lfw_eval.py#L24-L33) in groups(folds) of [600 entries](./lfw_eval.py#L27). R contains two images of the same identity [`.flag=1`](./lfw_eval.py#L28), L contains two images of different identity [`.flag=-1`](./lfw_eval.py#L33).

In the first step, it [creates features(embeddings)](./lfw_eval.py#L113-L115) for all [lfw](http://vis-www.cs.umass.edu/lfw/) images and [stores it](./lfw_eval.py#L128) in a temporary `.mat` file.

Then it [loads the result again](./lfw_eval.py#L62). For [the first](./lfw_eval.py#L63) [ten groups](./lfw_eval.py#L70) it [compares images as "scores"](./lfw_eval.py#L80) and then calculates a [`threshold`](./lfw_eval.py#L81) for the groups outside the fold by taking the threshold with [the highest accuracy](./lfw_eval.py#L55) found for `-1 ... 1` in [`1/10000`](./lfw_eval.py#L81) steps. Using the images within the fold we use [the re-asserted accuracy](./lfw_eval.py#L82).

The accuracy for a single group is evaluated following the [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) principle, in which - for a group of scores - it looks at the [found positives](./lfw_eval.py#L44) and the [found negatives](./lfw_eval.py#L45). The accuracy is derived following:

```python
(
    (found_positives / max_positives) + (found_negatives / max_negatives)
) / 2
```

The [mean of the accuracy for the first ten folds](./lfw_eval.py#L147) is used as final accuracy.
