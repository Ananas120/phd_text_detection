from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import csv
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc
from fiftyone.utils.eval.coco import COCODetectionResults

def read_predictions(model_names, dataset="new_custom_val_v2"):
  input_dir = os.path.join("/content/drive/MyDrive/Thesis/SSD/predictions",dataset)
  predictions_models = [[] for _ in model_names]
  for i, name in enumerate(model_names):
    # Define the name of the CSV file you want to read
    if dataset == "new_custom_val_v2":
      filename = "val_v2_big_" + name + ".csv"
    else:
      filename = name + ".csv"
    filename = os.path.join(input_dir, filename)
    # Open the CSV file in "read" mode
    with open(filename, mode="r", newline="") as csv_file:
        # Create a reader object using the csv library
        reader = csv.DictReader(csv_file)

        # Define an empty list to store the contents of the CSV file
        contents = []

        # Loop through each row in the CSV file
        for row in reader:
            row["image_id"] = int(row["image_id"])
            row["category_id"] = int(row["category_id"])
            row["bbox"] = eval(row["bbox"])
            row["score"] = float(row["score"])

            # Add the row to the contents list
            if len(row["bbox"]) == 4:
              contents.append(row)
        predictions_models[i] = contents
  return predictions_models

def save_results(model_names, val_dataset, dataset="new_custom_v2"):
  predictions_models = read_predictions(model_names, dataset=dataset) 

  for i, model_name in enumerate(model_names):
    eval_dir = "/content/drive/MyDrive/Thesis/SSD/evaluations/" + model_name

    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    if not os.path.exists(os.path.join(eval_dir, "results.json")):
      print(model_name)
      tmp = val_dataset.clone()
      classes = tmp.default_classes
      fouc.add_coco_labels(tmp, model_name, predictions_models[i], classes,coco_id_field="coco_id", label_type="detections")

      results = tmp.evaluate_detections(
        model_name,
        gt_field="detections",
        eval_key="eval_{}".format(model_name),
        compute_mAP=True,
      )

      results.write_json("/content/drive/MyDrive/Thesis/SSD/evaluations/{}/results.json".format(model_name))
      
      plot = results.plot_pr_curves(classes=["text"])
      plot.update_layout(title="{} - iou_thresh all".format(model_name))
      plot.write_image("/content/drive/MyDrive/Thesis/SSD/evaluations/{}/pr_avg.png".format(model_name))

      for thresh in [0.5,0.65,0.85]:
        plot = results.plot_pr_curves(classes=["text"], iou_thresh = thresh )
        plot.update_layout(title="{} - iou_thresh {}".format(model_name, thresh))
        plot.write_image("/content/drive/MyDrive/Thesis/SSD/evaluations/{}/pr_{}.png".format(model_name, int(100*thresh)))

      cm, _, _ = results._confusion_matrix(include_other = None, include_missing = None)

      disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ["Text", "Background"])
      disp.plot()

      plt.savefig("/content/drive/MyDrive/Thesis/SSD/evaluations/{}/confusion_matrix.png".format(model_name))

      tmp.delete()

def COCO_results(model_name, val_dataset):
    save_results([model_name], "V2_big", val_dataset)
    return rrc_evaluation_funcs.main_evaluation(p,default_evaluation_params,validate_data,evaluate_method, show_result = False)["method"]






