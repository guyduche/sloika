import cPickle
import json
json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

with open('model_labelled.pkl', 'r') as fh:
    model = cPickle.load(fh)


print json.dumps(model.json(), indent=4)
print json.dumps(model.json(True), indent=4)
