
![WhatsApp Image 2024-08-19 at 14 09 44](https://github.com/user-attachments/assets/8934eebd-1de8-4d34-9c5f-b9188fa12f10)

# Inter-Vechicular Depth Estimation from RGB Camera Feed


## About The Project

This project focuses on the estimation of headway, the distance between a vehicle and the vehicle directly in front, using depth estimation from monocular RGB video feeds. By leveraging deep learning techniques, the project aims to derive accurate distance measurements, which are crucial for understanding driving patterns in congested urban environments, particularly in Guwahati City.
### Prerequisites

The primary objectives of this research are:

1. Depth Estimation: Utilisation deep learning algorithms to extract depth information from monocular RGB video feeds, allowing for the estimation of distance to the vehicle ahead.

2. Headway Calculation: Derive the estimated headway between the host vehicle and the vehicle directly in front, providing essential data for analysing driving behaviour.

3. Behavorial Analysis: Integrate additional parameters such as driver speed, acceleration, and concentration, assessed through facial emotion detection, to develop a comprehensive understanding of driving patterns in highly congested areas.

4. Road Safety Improvement: Utilise the insights gained from the analysis to propose measures aimed at enhancing road safety and understanding driver responses to various stimuli.
## Usage

This research lays the groundwork for further studies in driver behavior and safety enhancements in urban settings. Future work may include:

1. Expanding the dataset to include various traffic conditions.
2. Enhancing the depth estimation model for increased accuracy in diverse environmental conditions.
3. Exploring the impact of environmental factors on driver behaviour.
## Getting Started

1. Clone repository.
2. Download proper model (based on environment of use) from [here](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models) and save in `assets/Depth_Anything_V2/checkpoints` folder.
3. Download test images/video for prediction into `test` folder.
4. Move to the base folder and run ``` pip install -r requirements.txt```. It is preferable to use a conda environment. Tested for python==3.11.5
5. The environment is ready to run. Use the following code in terminal:
> For Image:

```python run_image.py --input-path <path-to-input-directory> --outdir <path-to-output-directory> --encoder <vits/vitb/vitl/vitg> --max-depth <numeric-values-in-meters>```

> For Image:

```python run_video.py --input-path <path-to-input-directory> --outdir <path-to-output-directory> --encoder <vits/vitb/vitl/vitg> --max-depth <numeric-values-in-meters>```

Demo Result: [YouTube](https://www.youtube.com/watch?v=dbGtP_d9eHU)

Replace arguments with correct alternatives.
## Roadmap

- [x] Estimate depth in ambient lighting conditions
- [x] Detect for Cars using prebuilt Roboflow Model.
- [x] Develop custom model for multiple vehicle headway detection and increased detection accuracy.
- [x] Integration with Stereo-Camera data for validation figures (Ongoing)
- [ ] Integration with Driver Behaviour Model
## License

Distributed under the MIT License. See [MIT License](https://opensource.org/licenses/MIT) for more information.
## Contact

[Aditya Paul - Website](https://aditya-pauls-portfolio.vercel.app/) - [LinkedIn](https://www.linkedin.com/in/adityapaul03/) - adityapaul.official@outlook.com
