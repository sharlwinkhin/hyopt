# hyopt
This is the software artifacts used in the following paper:
Empirical Comparison of Hyperparameter Optimization Algorithms in the Context of Optimizing Deep Learning-based Malware Detectors


Scripts

	hyopt_ray.py : script for comparing hyperparam tuning algorithms on 7 datasets via Ray Tune platform
	optun_native.py : script for running optuna optimization library, without Ray Tune

Datasets

Maldroid (https://www.unb.ca/cic/datasets/maldroid-2020.html) : This dataset comprises  static and dynamic features that characterize whether an Android app belongs to a malware family (Adware, Banking malware, SMS malware, or Riskware) or is a benign app. The features represent information about use of permissions, sensitive APIs, system calls, frequency counts for different file types incidents of obfuscation, etc. Statically extracted information such as intents, permissions and services, frequency counts for different file types, incidents of obfuscation, and sensitive API invocations are used. Dynamically observed behaviors such as system calls, binder calls and composite behaviors are also used. The target variable is a multi-class variable ranging from 0 to 5. 

PDFMal (https://www.unb.ca/cic/datasets/pdfmal-2022.html) : This dataset comprises static, use- and frequency-type features, such as use of encryption, number of embedded files, number of Xref entries, and number of specific keywords (e.g., `JavaScript'), which can be used to identify malicious PDF files. It contains both use- and frequency-types of features such as use of encryption, number of embedded files, number of Xref entries, etc. The target variable is a binary class --- 1 or malware and 0 for benign.

CryptoI and CryptoII (https://github.com/cslfiu/IoTCryptojacking) : These two datasets comprise statistical features computed from time-series data of network packets sent by a Cryptojacking malware infected Raspberry Pi device and a Cryptojacking malware infected desktop. 
	
LR-Dos (B. Tushir, Y. Liu, and B. Dezfouli, “Leveraging frame aggregation in wi-fi iot networks for low-rate ddos attack detection,” in Network and System Security: 16th International Conference, NSS 2022, pp. 319–334) : This dataset comprises statistical features, such as density of network packets and communication channel utilization, which can be used to detect low rate denial of service attacks to smart home devices.

PermSeq and PackageFreq (https://github.com/sharlwinkhin/msoft20): These two datasets comprise static features that characterize whether an Android app is a malware or not. The first one contains features represent the sequence of Android API calls at class level, e.g., app.Dialog followed by telephony.SmsMessage. The second one contains features that represent the frequency of Android API invocations. The target variable is a binary class for both datasets.

