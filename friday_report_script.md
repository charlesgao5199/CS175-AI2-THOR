# Friday Report Script

Project: Explicit vs. Implicit Memory and Semantic Priors in Object Goal Navigation

Suggested pace: each person speaks for about 1 to 1.5 minutes.

## Speaker 1: Charles Gao

Hello Professor Kask. We are Group 66. Our project is about object-goal navigation in AI2-THOR and ProcTHOR.

The task is: given a target object, such as an apple, mug, or microwave, the agent must move through a 3D house and stop when it finds the object.

Our main question is whether explicit structure helps navigation. We compare three methods: end-to-end reinforcement learning, semantic mapping with classical planning, and semantic mapping with LLM guidance.

Method 1 is our end-to-end RL baseline. It uses RGB-D images, the target category, and compass information. A ResNet encodes the image, a GRU provides memory, and PPO trains the policy.

In this method, both memory and object knowledge are implicit. They are hidden inside the model. This gives us a standard black-box baseline to compare against the more structured methods.

## Speaker 2: Emily Sun

I will explain the input, output, and semantic map.

At each step, the agent receives an RGB image, a depth image, compass information, and a target object category. It chooses one action: move forward, turn, look up or down, or stop.

For Methods 2 and 3, we build an explicit top-down semantic map. The map stores explored areas, walkable space, and detected object classes.

To build the map, we detect objects in the RGB image, use depth to project them into the scene, and update a 2D grid. Over time, the map becomes the agent's memory.

This helps the agent avoid searching the same place again. It also helps us visualize what the agent has seen and where it has moved.

## Speaker 3: Junyuan Zhang

I will explain Method 2 and evaluation.

Method 2 uses semantic mapping plus classical planning. If the target object appears on the map, the agent uses A star search to move toward it and then stops.

If the target has not been found, the agent uses frontier exploration. A frontier is the boundary between explored and unexplored space. The agent moves toward frontiers to search new areas.

This method has explicit memory because it keeps a map. But it does not use an LLM, so it has only simple common sense.

For evaluation, we will test all methods with the same scenes and seeds. Our main metrics are Success Rate, SPL, and SoftSPL. These measure whether the agent succeeds and how efficient its path is.

We will also compare runtime, generalization, per-object performance, and failure cases.

## Speaker 4: Tong Zhao

I will explain Method 3 and our expectations.

Method 3 uses the same semantic map as Method 2, but adds LLM guidance. Every few steps, we turn the map into a short text summary and ask the LLM which region seems most promising.

For example, if the target is a microwave and one region looks like a kitchen, the LLM may suggest exploring that region first. The LLM gives a high-level choice, and the planner handles the actual movement.

This method combines explicit memory from the map with explicit common sense from the language model.

We expect the random baseline to be weak. Method 1 should improve with training, but may need more data. Method 2 should be more stable because it remembers explored areas. Method 3 may work best when object-location common sense matters.

If the full setup is too unstable, our backup plan is to use a simpler iTHOR setup while keeping the same comparison. Overall, we want to compare not only performance, but also why each method succeeds or fails. Thank you.

## Very Short Q&A

Question: What is the main difference between Method 2 and Method 3?

Answer: Method 2 uses frontier exploration. Method 3 uses an LLM to choose more promising regions.

Question: What does success mean?

Answer: The agent stops when the target is visible and close enough.

Question: What is the biggest risk?

Answer: Simulator and training stability. Our backup is a simpler iTHOR setup.
