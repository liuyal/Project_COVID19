# Project COIVD-19

### A Data Mining Study on COVID-19 Pandemic Growth & Related Social Media Dynamics 

## Introduction
Coronavirus disease 2019 or COVID-19 is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) [1]. First identified back in December 2019 in Wuhan province, China, the COVID-19 virus has resulted in nearly 15 million confirmed cases globally (as the writing of this report) [2]. Amidst the COVID-19 crisis, social media usage on platforms such as Facebook, WhatsApp, Twitter, and etc. has surged significantly [3]. As the general population rely heavily on social media platforms to gather the latest information in regards to the pandemic, resulting in an unprecedented amount of content and information.

An interesting data mining topic to focus on for the COVID-19 pandemic is to determine the relationship between COVID-19 related trending topics and sentiments on the social media platform Twitter, with the number of reported confirmed cases for a given country over a period of time. Topic modeling and Tweet sentiments classification could be a useful measurement for determining the general attitude expressed towards COVID-19 for a given population. Since the pandemic originated from Asia, and only after a three-month period where the virus quickly spread across North America totaling the number of confirmed cases to more than 10 million globally [2]; it would be very interesting to follow the change in daily trending topics of interest and tweet sentiments due to the influx of confirmed cases as COVID-19 begins to spread in a particular country or region.

The measurement of relationship between the growth of the pandemic and social media topics and semantics can help determine how a general population express their opinions, concerns, and general awareness throughout a global event. As such, semantics information and topic modeling can be used by government bodies around the world to determine the general population’s level of attitude towards the pandemic as it grows, and how to issue proper procedures and implement restriction in times of crisis for future global events or pandemics.

## Architecture & Pipeline

![](documents/assets/DM_Architecture.png)

The data mining architecture follows the basic Knowledge Discovery and Data Mining (KDDM) process, where the entire process is split into five phase. Staring with data collection for obtaining the raw datasets using the data collector, then data preprocessing using the data formatter to prepare the datasets for transformation such as Tokenization. Data mining methods such as sentiment classifiers and topic modeling are then performed on the transformed dataset and the results are visualized for knowledge comprehension.

## Results

![](documents/assets/chart.PNG) 

The first spike of negative sentiment tweets are near the end of the month of February, where the COVID-19 growth starts to manifest rapidly in areas outside of China. The second spike from mid-May onwards, which can be correlated to the explosive growth within the United States. As the number of confirmed cases increased from around 1.5 million to 4.5 million in the span of two months, so did the frequency of negative sentiment tweets. The day to day increase in number of negative sentiment tweets can be observed as the severity and number of confirmed cases increased due to COVID-19 pandemic.

The full report can be found [HERE](documents/report.pdf)

## How to Run
 - Input Twitter API Credentials to twitter.token ([GUIDE](https://projects.raspberrypi.org/en/projects/getting-started-with-the-twitter-api/2))
 - Run master script with python 3.6+ `python 0_project_covid19.py`

## Reference
- [1] M. Clinic, “Coronavirus disease 2019 (COVID-19),” Mayo Clinic, 16-Jun-2020. [Online]. Available: https://www.mayoclinic.org/diseases-conditions/coronavirus/symptoms-causes/syc-20479963. [Accessed: 25-Jun-2020].
- [2] L. Gardner, “Mapping COVID-19,” JHU CSSE. [Online]. Available: https://systems.jhu.edu/research/public-health/ncov/. [Accessed: 25-Jun-2020].
- [3] R. Holmes, “Is COVID-19 Social Media's Levelling Up Moment?” Forbes, 24-Apr-2020. [Online]. Available: https://www.forbes.com/sites/ryanholmes/2020/04/24/is-covid-19-social-medias-levelling-up-moment/#93725e96c606. [Accessed: 25-Jun-2020].
- [4] E. Chen, K. Lerman, E. Ferrara, I. S. Institute, C. A. C. C. A. E. Ferrara, C. Author, C. C. A. E. Ferrara, Close, and L. authors..., “Tracking Social Media Discourse About the COVID-19 Pandemic: Development of a Public Coronavirus Twitter Data Set,” JMIR Public Health and Surveillance. [Online]. Available: https://publichealth.jmir.org/2020/2/e19273/. [Accessed: 02-Jul-2020].

## Dataset Links
- [Johns Hopkins University (CSSE) COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19)
- [Daily COVID-19 Twitter ID Data Repository](https://github.com/echen102/COVID-19-TweetIDs)
- [T4SA Twitter Training Data](http://www.t4sa.it/)
