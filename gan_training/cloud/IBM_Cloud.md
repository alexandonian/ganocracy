# Creating Virtual Machines on IBM Cloud

Training GANs for image synthesis can be computationally expensive. In reality, one or several modern GPUs are needed to train GANs in a reasonable time frame. If you don't have access to such resources, it is possible train your GANs in the "cloud," taking advantage of GPU virtual machine instances offered by the various cloud providers. This document will walk through creating appropriate virtual machines on IBM cloud.

### Step 1: Create an account on IBM Cloud.

Navigate to [IBM Cloud](https://www.ibm.com/cloud/) and create an account (or log into an existing account) by following the on-screen instructions. If you are opening a new account, you will likely need to enable billing to support your cloud usage. Once you have reached your IBM cloud dashboard, navigate to the the `Manage -> Billing and usage` tab in the top right hand corner begin the process. We leave the rest of this step as an exercise to the reader.

### Step 2: Create a resource.

1. On your dashboard, navigate and click on  `Create resource` in the top-right corner.
2. Select the type of resource you would like to create. In this tutorial, we will be creating and `Virtual Server` (as shown), since they are sufficiently powerful and more conveniently administered. You could also choose to provision a `Bare Metal Server` if you find you need more configuration flexibilty, but these typically take longer to create (about a day or so) and tend to be more expensive.
3. If moving forward with a virtual server, then read the different  descriptions and select the appropriate type based on your specific requirements. In this tutorial, we will select `Public-Virtual Server`, as it strikes a good balance between cost, commitment and stability.
4. Configure your instance to suit your needs and budget.  We recommend some variation on the following:

**Location:** If you concerned about maintaining low-latency connections to your machine, select the region closest to your current location. Furthermore, if you are planning on creating more than one instance and need low-latency inter-node communication (e.g. as needed with distributed training), try to create all of your instances in the same region. (*Note:* Some regions may not offer the desired server profiles. If so, try looking in the next closest region). Here, we choose the `NA East` location.

**Profiles:** It is *highly* recommended that you select a profile with GPUs. Click `All profiles` and then select the `GPU` tab. You will see a list of the available GPU instance. NVIDIA P100 and V100 GPUs are high performance, datacenter-class GPUs with 16GB of vRAM well suited for running deep learning experiments. We opt for the `AC2.16x120` instance with 2xV100 GPUs, 16 vCPUs, 120GB of RAM, which has a base price of $4.437/hour at the current time of writing.

**SSH keys (optional, but recommended):** Add your SSH keys profile from the drop down list if you would like to be able to log into your machine without entering your password each time. This is particularly convenient if you are creating many machines and/or would like to run automated deployment/setup scripts in bulk. For more information about how to add SSH keys, see [INSERT INSTRUCTIONS].

**Image:** You are free to choose any OS image of your liking, but we recommend some flavor of Ubuntu (16.04 or greater) as it provides excellent compatability with many deep learning libraries and projects.

**Attached storage disks:** The boot disk's primary purpose is to store the operating system, so it needed be that large. For small scale experimentation, choosing a larger boot disk may be all that is needed. If you are planning on training with large-scale datasets, it is recommended that you add additional disks to store the data. (*Note:* If you add additional disks, you will need to format and mount them to your instance so that they can be used. See Step 4 for more details).

**Network Interface:** Here, you can configure various network speicifications. If you will be doing large amounts of data transfer/communication with other nodes, we recommend upgrading to the 1 Gbps Public & Private Network Uplinks. Otherwise, the provided defaults should be sufficient.


### Step 3: Connecting to your resource.
[TODO]

### Step 4: Prepare your resource for experiments.

Depending on your configuration, you may have to do a few last steps before your machine is ready to run deep learning experiments. For more information about installing required software, mounting data disks, and transfering data, please see [INSERT LINK TO SETUP COMMON INSTRUCTIONS].