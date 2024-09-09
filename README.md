## PyTorch single-node multi-GPU example for ASC's supercomputers Triton

To make usage of [PyTorch distributed][pytorch_dist], in particular
[DistributedDataParallel][ddp] with a large dataset, like ImageNet on Triton easier, we created this example. This work is ongoing, and we welcome any feedback to improve it.

**PyTorch distributed** offers a nice way of running multi-GPU and multi-node PyTorch jobs. (Triton only supports single-node, multi-GPU jobs for the time being). Unfortunately, The official PyTorch
documentation has been a bit lacking in this area, and online examples are not always directly applicable to Triton's environment. 

Note: in this example we are using Slurm's `srun` to launch multiple processess on one GPU node.

**Using ImageNet**: For training models with large datasets like ImageNet on a cluster environment like Triton, we recommend extracting images from pre-downloaded compressed datasets (e.g., /scratch/shareddata/dldata/imagenet/ILSVRC2012_img_train.tar) to the [local drives](https://scicomp.aalto.fi/triton/usage/localstorage/) of the compute node you're using, rather than to your `WRKDIR`. Here's why:

- **Shared Storage Performance:** The /scratch/work is shared over all users making per-user I/O performance actually rather poor, so repeatedly reading a dataset from it should be avoided. GPU nodes on Triton are equipped with SSD-raids, which provide individual I/O speeds of up to 30GB/s from the local filesystem (e.g., /tmp). This makes data loading much faster when using local storage.
- **Lustre File System Considerations:** Triton, like many clusters, uses the Lustre network file system, which is optimized for large files. It performs poorly with smaller files (less than 10MB). Large datasets containing many small files (<1MB) incur significant network overhead, which can slow down your data processing and impact system performance for other users. For more information, refer to our docs: https://scicomp.aalto.fi/triton/usage/lustre/#storage-lustre-scratch. 
- **Save your quota.**

If you want to do some interactive exploration of the dataset, such as displaying some images, trying out various image transformations, you can extract a few classes to the WRKDIR, see the script `extract_some_samples.sh` 

The model training will be run normally, but forwarding all the output to local disk is recommended and in the end copying relevant output to `WRKDIR` for analysis and further usage.

For more advanced usage of ImageNet, checkout [this repo](https://github.com/AaltoRSE/ImageNetTools).

If you are working on CSC's supercomputers that support multi-node, multi-GPU jobs, you can refer to CSC's [multi-node ddp examples](https://github.com/CSCfi/pytorch-ddp-examples).


[pytorch_dist]: https://pytorch.org/tutorials/beginner/dist_overview.html
[ddp]: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
