<template>
  <div class="imageEditorApp">
    <header>
      <h1>MedTextCleaner</h1>
      <div class="button-group">
        <button class="redact-button" @click="redact">Download DICOM</button>
      </div>
    </header>
    <tui-image-editor
      ref="tuiImageEditor"
      :include-ui="useDefaultUI"
      :options="options"
      @objectAdded="onObjectAdded"
    >
    </tui-image-editor>
    <div class="spinner" v-show="loading"></div>
  </div>
</template>
<script>
import { ImageEditor } from "@toast-ui/vue-image-editor";
import axios from "axios";


export default {
  components: {
    "tui-image-editor": ImageEditor,
  },
  data() {
    return {
      useDefaultUI: true,
      options: {
        includeUI: {
          loadImage: {
            path: "background.jpeg",
            name: "Test",
          },
          menu: ['shape'],
          menuBarPosition: "right",
        },
        usageStatistics: false,
      },
      loading: false,
      file: null,
      rectIds: [],
    };
  },
  created() {
    this.loading = true; // Show the spinner
    this.rectIds = []
    const urlParams = new URLSearchParams(window.location.search);
    this.instance = urlParams.get('uuid');
    const url = "http://localhost:8000/instances/" + this.instance + "/preview"
    fetch(url)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.blob();
        })
      .then(async (imageBlob) => {
        this.$refs.tuiImageEditor
              .invoke("loadImageFromFile", imageBlob, this.instance)
              .catch((error) => {
                  console.error("Error loading image from file: ", error);
                });
        axios
          .post("http://localhost:8000/predict", imageBlob, {
            headers: {
              "Content-Type": "image/jpeg",
            },
          })
          .then(async (response) => {
            const boxes = response.data.boxes
            for (const elem of boxes) {
              var props = await this.$refs.tuiImageEditor.invoke("addShape", "rect", {
                      fill: "",
                      stroke: "#ffbb3b",
                      strokeWidth: 3,
                      width: elem[2],
                      height: elem[3],
                      left: Math.ceil(elem[0] + elem[2]/2),
                      top: Math.ceil(elem[1] + elem[3]/2),
                    });
              this.rectIds.push(props.id)
            }
          })
          .catch((error) => {
            console.error("Error sending image to server:", error);
          })
          .finally(() => {
            this.loading = false; // Hide the spinner
          });
      })
      .catch((error) => {
        console.error('There was a problem with the fetch operation:', error);
      })
  },
  mounted() {
    setTimeout(() => {
      const resetButton = document.querySelector('.tie-btn-reset');
      if (resetButton) {
        resetButton.style.display = 'none';
      }
    }, 50); // You may need to adjust the timeout duration
  },
  methods: {
    onObjectAdded(props) {this.rectIds.push(props.id)},
    redact() {
      this.loading = true; // Show the spinner
      const rectangles = []
      for (const id of this.rectIds) {
        const rect = this.$refs.tuiImageEditor.invoke("getObjectProperties",id, ['left', 'top', 'width', 'height', 'strokeWidth']);
        if (rect) {
          const top = Math.floor(rect.top - rect.height/2)
          const left = Math.floor(rect.left - rect.width/2)
          const width = Math.ceil(rect.width + rect.strokeWidth)
          const height = Math.ceil(rect.height + rect.strokeWidth)
          rectangles.push([left, top, width, height])
        }
      }
      
      const data = {
        rectangles: rectangles
      };
      axios
        .post("http://localhost:8000/redact/"+this.instance, JSON.stringify(data), {
          headers: {
            "Content-Type": "application/json",
          },
        })
        .then((response) => {
          window.location.href = `http://localhost:8000/app/explorer.html#instance?uuid=${response.data}`;
        })
        .catch((error) => {
          console.log("Error downloading data:", error);
        })
        .finally(() => {
            this.loading = false; // Hide the spinner

        });
    }
  },
};
</script>


<style>
body {
  margin: 0;
  padding: 0;
  background-color: #141414; /* Set the background color to black */
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px; /* Add some padding to the sides of the header */
}

h1 {
  color: #FFBF3C; /* Set the text color of the header */
}

.button-group {
  display: flex;
  align-items: center;
  margin-right: 5px;
}

button {
  font-size: 11px;
  border-radius: 20px;
  font-weight: bold;
  border: none;
  padding: 9px 28px;
  margin: 0px 4px;
  cursor: pointer;
  outline: none;
  transition: 0.3s;
}


button.redact-button {
  background-color: #FFBF3C;
  color: #FFFFFF;
}

button:hover {
  opacity: 0.8;
}

.spinner {
  position: fixed;
  left: 50%;
  top: 50%;
  width: 50px;
  height: 50px;
  margin-left: -50px;
  margin-top: -25px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #FFBF3C;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>



<style scoped>
.imageEditorApp {
  width: 100vw;
  height: calc(100vh - 60px); /* Increase the header height subtraction to avoid scrollbar */
  display: flex;
  flex-direction: column;
}
h1 {
  margin: 10px;
}
</style>


