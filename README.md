# hyopt
This is the software artifacts used in the following paper:
Empirical Comparison of Hyperparameter Optimization Algorithms in the Context of Optimizing Deep Learning-based Malware Detectors


Scripts

	hyopt_ray.py : script for comparing hyperparam tuning algorithms on 7 datasets via Ray Tune platform
	optun_native.py : script for running optuna optimization library, without Ray Tune

Datasets

	Maldroid: https://www.unb.ca/cic/datasets/maldroid-2020.html
	PDFMal: https://www.unb.ca/cic/datasets/pdfmal-2022.html
	CryptoI and CryptoII: https://github.com/cslfiu/IoTCryptojacking
	LR-Dos: . B. Tushir, Y. Liu, and B. Dezfouli, “Leveraging frame aggregation in wi-fi iot networks for low-rate ddos attack detection,” in Network and System Security: 16th International Conference, NSS 2022, pp. 319–334.

