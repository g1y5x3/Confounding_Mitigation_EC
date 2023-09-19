import wandb

api = wandb.Api()

run = api.run("g1y5x3/LOO-Sentence-Classification/q2xz3f4w")
run.summary["exp_info/healthy_samples"]   = 2014
run.summary["exp_info/fatigued_samples"]  = 4191
run.summary["exp_info/total_samples"]     = 2177
run.summary["metrics/train_acc"]          = 0.9836
run.summary["metrics/test_acc"]           = 1
run.summary["subject_info/vfi_1"]         = 0

run.summary.update()