import numpy as np
from perception.confidence_estimator import ConfidenceEstimator
ce=ConfidenceEstimator()
fake_depth=np.random.rand(480,640).astype('float32')
conf=ce.estimate(fake_depth)
print('confidence shape:',conf.shape)
print('mean confidence:', round(ce.mean_confidence(conf),4))
print('regions:',ce.regional_confidence(conf))
print('success')