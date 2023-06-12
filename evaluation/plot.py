import plotly.express as px
import plotly.graph_objects as go
import eta.core.utils as etau
import numpy as np
import fiftyone as fo
from fiftyone.utils.eval.coco import COCODetectionResults

_DEFAULT_LAYOUT = dict(
    template="ggplot2", margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

ref_iou_threshs = np.arange(0.5,1,0.05)

def _get_iou_thresh_inds(iou_thresh=None):
  if iou_thresh is None:
      return np.arange(len(ref_iou_threshs))

  if etau.is_numeric(iou_thresh):
      iou_threshs = [iou_thresh]
  else:
      iou_threshs = iou_thresh

  thresh_inds = []
  for iou_thresh in iou_threshs:
    found = False
    for idx, ref_iou_thresh in enumerate(ref_iou_threshs):
      if np.abs(float(iou_thresh) - ref_iou_thresh) < 1e-6:
        thresh_inds.append(idx)
        found = True
        break
    if not found:
      raise ValueError(
          "Invalid IoU threshold %f. Refer to `results.iou_threshs` "
          "to see the available values" % iou_thresh
      )

  return thresh_inds


def _get_qualitative_colors(num_classes, colors=None):
    # Some color choices:
    # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
    if colors is None:
        if num_classes == 1:
            colors = ["#FF6D04"]
        elif num_classes <= 10:
            colors = px.colors.qualitative.G10
        else:
            colors = px.colors.qualitative.Alphabet
    colors = list(colors)
    return [colors[i % len(colors)] for i in range(num_classes)]

def plot_pr_curves(
    results,
    model_names,
    iou_thresh=None,
    figure=None,
    title=None,
    **kwargs,
):
    """Plots a set of per-class precision-recall (PR) curves.
    Args:
        precisions: a ``num_classes x num_recalls`` array-like of per-class
            precision values
        recall: an array-like of recall values
        classes: the list of classes
        thresholds (None): a ``num_classes x num_recalls`` array-like of
            decision thresholds
        figure (None): a :class:`plotly:plotly.graph_objects.Figure` to which
            to add the plots
        title (None): a title for the plot
        **kwargs: optional keyword arguments for
            :meth:`plotly:plotly.graph_objects.Figure.update_layout`
    Returns:
        one of the following
        -   a :class:`PlotlyNotebookPlot`, if you are working in a Jupyter
            notebook
        -   a plotly figure, otherwise
    """
    thresh_inds = _get_iou_thresh_inds(iou_thresh=iou_thresh)

    precisions = []
    recalls = []
    thresholds = []

    for i in range(len(model_names)):
        class_id = 0
        precisions.append(
            np.mean(results[i].precision[thresh_inds, class_id], axis=0)
        )
        thresholds.append(
            np.mean(results[i].thresholds[thresh_inds, class_id], axis=0)
        )
        recalls.append(results[i].recall)

    if figure is None:
        figure = go.Figure()

    # Add 50/50 line
    figure.add_shape(
        type="line", line=dict(dash="dash"), x0=0, x1=1, y0=1, y1=0
    )

    hover_lines = [
        "<b>model: %{text}</b>",
        "recall: %{x:.3f}",
        "precision: %{y:.3f}",
    ]

    if thresholds is not None:
        hover_lines.append("threshold: %{customdata:.3f}")

    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    # Plot in descending order of AP
    avg_precisions = np.mean(precisions, axis=1)
    inds = np.argsort(-avg_precisions)  # negative for descending order

    colors = _get_qualitative_colors(len(inds))

    for idx, color in zip(inds, colors):
        precision = precisions[idx]
        recall = recalls[idx]
        _model = model_names[idx]
        avg_precision = avg_precisions[idx]
        label = "%s (AP = %.3f)" % (_model, avg_precision)

        if thresholds is not None:
            customdata = thresholds[idx]
        else:
            customdata = None

        line = go.Scatter(
            x=recall,
            y=precision,
            name=label,
            mode="lines",
            line_color=color,
            text=np.full(recall.shape, _model),
            hovertemplate=hovertemplate,
            customdata=customdata,
        )

        figure.add_trace(line)

    figure.update_layout(
        xaxis=dict(range=[0, 1], constrain="domain"),
        yaxis=dict(
            range=[0, 1], constrain="domain", scaleanchor="x", scaleratio=1
        ),
        xaxis_title="Recall",
        yaxis_title="Precision",
        title=title,
    )

    figure.update_layout(**_DEFAULT_LAYOUT)
    figure.update_layout(**kwargs)


    return figure




def get_run_config(model_name):
  run_config = """{{
    "method": "coco",
    "cls": "fiftyone.utils.eval.coco.COCOEvaluationConfig",
    "pred_field": "{}",
    "gt_field": "detections",
    "iou": 0.5,
    "classwise": true,
    "iscrowd": "iscrowd",
    "use_masks": false,
    "use_boxes": false,
    "tolerance": null,
    "compute_mAP": true,
    "iou_threshs": [
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95
    ],
    "max_preds": 100,
    "error_level": 1
}}""".format(model_name)
  return run_config

def get_eval_key(model_name):
  eval_key = "eval_{}".format(model_name)
  if model_name == "Textboxes":
    eval_key = "eval"
  return eval_key


def read_results(model_names, dataset):
  results = []
  for model_name in model_names:
    run_config = get_run_config(model_name)
    eval_key = get_eval_key(model_name)
    res = COCODetectionResults.from_json(
      "/content/drive/MyDrive/Thesis/SSD/evaluations/{}/results.json".format(model_name),
      samples = dataset,
      config = fo.core.runs.RunConfig.from_str(run_config),
      key = eval_key
      )
    results.append(res)
  return results