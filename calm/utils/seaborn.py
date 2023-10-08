import seaborn as sns
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embedding(data, title):
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)
	fig = plt.figure()
	ax = plt.subplot(111)
	for i in range(data.shape[0]):
		plt.plot(data[i, 0], data[i, 1], 'o')
	plt.xticks()
	plt.yticks()
	plt.title(title, fontsize=14)
	return fig

def test_reward(file_path):
	data = 0
	count = 0
	for line in open(file_path, "r"):
		data += float(line)
		count += 1

	print("ave reward = ", data/count)





if __name__ == "__main__":
	# array1 = np.load("/home/cjm/CALM/output/clipfeature/03-14/motion_feature_location.npy", allow_pickle=True)
	# data = enumerate(array1.flatten())
	# print(0)

	array1 = np.load("/home/cjm/CALM/output/sim_location.npy", allow_pickle=True)
	data = array1.tolist()
	frames = np.array(list(data.keys()))
	rewards = list(data.values())
	for i in range(len(rewards)):
		rewards[i] = rewards[i].item()
	plt.plot(frames,rewards)
	plt.show()

    # sns.set()
    # sns.set_style("darkgrid")
    # array1 = np.load("/home/cjm/CALM/output/sim_8.npy", allow_pickle=True)
    # ts = TSNE(n_components=2, init='pca', random_state=0)
    # result = ts.fit_transform(array1)
    # fig = plot_embedding(result, 't-SNE Embedding of digits')
    # plt.show()

	# test_reward("/home/cjm/CALM/output/disc_reward.txt")
	# test_reward("/home/cjm/CALM/output/cdisc_reward.txt")

	# array1 = np.load("/home/cjm/CALM/output/motion_feature.npy", allow_pickle=True)
	# array2 = np.load("/home/cjm/CALM/output/text_feature.npy", allow_pickle=True)
	# data = array1.tolist()
	# frames = np.array(list(data.keys()))
	# rewards = list(data.values())
	#
	# for i in range(len(rewards)):
	# 	rewards[i] = np.exp((rewards[i].item())*0.15)
	# plt.plot(frames,rewards)
	# # sns.lineplot(x=frames, y=rewards)
	# plt.show()
