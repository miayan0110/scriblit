import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random
from PIL import Image, ImageFilter
import numpy as np

from datasets import load_from_disk, Image as HfImage, Dataset as DS
from physical_relighting_api_v2 import compute_relighting, PhysicalRelightingConfig

class Indoor_dataset(Dataset):
	def __init__(self, tokenizer, ds_path):
		self.tokenizer = tokenizer
		self.ds: DS = load_from_disk(ds_path)
		print(self.ds)
		for col in self.ds.column_names:
			if col not in ('color', 'intensity', 'prompt'):
				self.ds = self.ds.cast_column(col, HfImage(decode=True))

		self.image_transforms = transforms.Compose([
			transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
		])

		self.conditioning_image_transforms = transforms.Compose([
			transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
			transforms.ToTensor(),
		])
		self.transform = transforms.ToTensor()
		self.colors = torch.tensor([
			[255, 255, 255],  # white
			[255, 0, 0],      # red
			[0, 255, 0],      # green
			[0, 0, 255],      # blue
			[255, 255, 0],    # yellow
			[255, 165, 0],    # orange
			[128, 0, 128],    # purple
			[255, 192, 203],  # pink
			[0, 255, 255],    # cyan
			[255, 0, 255]     # magenta
		], dtype=torch.float32)
		self.colors = self.colors / 255.0
  
		self.intensity = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], dtype=torch.float32)
		
	def __len__(self):
		return len(self.ds)

	def __getitem__(self, idx):
		item = self.ds[idx]
		
		if np.array(item['lightmap']).max() != 0:
			# for scribblelight
			ori = item['image']
			albedo = item['albedo']
			normal = item['normal']
			prompt = item['prompt']

			# relight by physical formula
			depth = item['depth'].convert('L')
			mask = item['mask'].convert('L')
			# intensity = item['intensity'][-1]
			# color = item['color']
			intensity = self.intensity[4]
			color = self.colors[idx % len(self.colors)]
			config = PhysicalRelightingConfig(ori, normal, depth)
			config.add_mask(mask, color, intensity)
			i_phys, lightmap = compute_relighting(config)
			# i_phys = item['pseudo_gt']
			# lightmap = item['lightmap']

			# scribble = shading2scrib(lightmap)
			# lightmap = lightmap.convert('RGB')
			# scribble = lightmap.convert('RGB')	# ex2_4 之前都是用這個

			# Apply transforms
			normal = self.conditioning_image_transforms(normal)
			depth = self.conditioning_image_transforms(depth)
			mask = self.conditioning_image_transforms(mask)
			lightmap = self.conditioning_image_transforms(lightmap)
			# scribble = self.conditioning_image_transforms(scribble)	# ex2_4 之前都是用這個
			ori = self.image_transforms(ori)
			albedo = self.image_transforms(albedo)
			i_phys = self.image_transforms(i_phys)

			color = torch.tensor(color, dtype=torch.float32)
			intensity = torch.tensor([intensity], dtype=torch.float32)
			
			# source = shading
			# source = torch.cat((normal, scribble), dim=0)	# ex2_4 之前都是用這個
			# ex2_5
			DM = depth * mask
			source = torch.cat((normal, depth, mask, DM), dim=0)

			# control_target = torch.cat((normal, lightmap), dim=0)	# ex2_4 之前都是用這個
			control_target = torch.cat((normal, depth, mask, DM), dim=0)	# ex2_5

			prompt = self.tokenizer(
				prompt, max_length=self.tokenizer.model_max_length-4, padding="max_length", truncation=True, return_tensors="pt"
			)

			return dict(pixel_values=i_phys, input_ids=prompt.input_ids, albedo=albedo, intensity=intensity, color=color, conditioning_pixel_values=source, control_target=control_target, ori_img=ori, normal=normal, depth=depth, mask=mask)


class Lightmap_dataset(Dataset):
	def __init__(self, tokenizer, ds_path):
		self.tokenizer = tokenizer
		self.ds: DS = load_from_disk(ds_path)
		print(self.ds)
		for col in self.ds.column_names:
			if col not in ('color', 'intensity'):
				self.ds = self.ds.cast_column(col, HfImage(decode=True))

		self.conditioning_image_transforms = transforms.Compose([
			transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
			transforms.ToTensor(),
		])
		self.transform = transforms.ToTensor()
		
	def __len__(self):
		return len(self.ds)

	def __getitem__(self, idx):
		item = self.ds[idx]
		
		if np.array(item['lightmap']).max() != 0:
			normal = item['normal']
			depth = item['depth']
			mask = item['mask']
			lightmap = item['lightmap']

			# Apply transforms
			normal = self.conditioning_image_transforms(normal)
			depth = self.conditioning_image_transforms(depth)
			mask = self.conditioning_image_transforms(mask)
			lightmap = self.conditioning_image_transforms(lightmap)

			control_source = torch.cat((mask, normal, depth), dim=0)

			return dict(control_source=control_source, control_target=lightmap)


def shading2scrib(image):
	odd_numbers = list(range(3, 20, 2))
	random_odd = random.choice(odd_numbers)

	image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
	image_np = np.array(image)

	#might need to change the parameter a & c
	a = 180
	c = 100

	output_image_np = np.zeros_like(image_np)
	output_image_np[image_np > a] = 255
	output_image_np[image_np < c] = 0
	output_image_np[(image_np <= a) & (image_np >= c)] = 127

	output_image = Image.fromarray(output_image_np)

	dilated_image = output_image.filter(ImageFilter.MinFilter(random_odd))
	eroded_image = dilated_image.filter(ImageFilter.MaxFilter(random_odd))
	return eroded_image