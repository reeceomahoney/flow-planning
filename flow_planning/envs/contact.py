"""Per-world detection of contact with the shape labelled "obstacle"."""

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def obstacle_contact_kernel(
    contact_count: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    shape_is_obstacle: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    # outputs
    contacted: wp.array(dtype=wp.int32),
    contact_shape: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    s0 = contact_shape0[tid]
    s1 = contact_shape1[tid]
    if s0 < 0 or s1 < 0:
        return
    if shape_is_obstacle[s0] == shape_is_obstacle[s1]:
        return

    x0 = wp.transform_identity()
    b0 = shape_body[s0]
    if b0 >= 0:
        x0 = body_q[b0]
    x1 = wp.transform_identity()
    b1 = shape_body[s1]
    if b1 >= 0:
        x1 = body_q[b1]
    p0 = wp.transform_point(x0, contact_point0[tid])
    p1 = wp.transform_point(x1, contact_point1[tid])
    gap = wp.dot(contact_normal[tid], p1 - p0)
    gap -= contact_margin0[tid] + contact_margin1[tid]
    if gap >= 0.0:
        return

    s = s0
    if shape_is_obstacle[s0] == 1:
        s = s1
    w = shape_world[s]
    if contacted[w] == 0:
        contact_shape[w] = s
    contacted[w] = 1


class ObstacleContactSensor:
    """Per-world latch set while anything is touching the obstacle."""

    def __init__(self, model):
        is_obstacle = np.array(
            [label.split("/")[-1] == "obstacle" for label in model.shape_label],
            np.int32,
        )
        self.model = model
        sbody = model.shape_body.numpy()
        bkey = list(getattr(model, "body_label", []) or [])
        self.labels = []
        for s, lab in enumerate(model.shape_label):
            b = int(sbody[s])
            name = str(bkey[b]).split("/")[-1] if 0 <= b < len(bkey) else ""
            if not name or name.startswith("body_"):
                name = str(lab).split("/")[-1]
            self.labels.append(name)
        assert not all(n.startswith("shape_") for n in self.labels), (
            "contact labels fell back to shape indices; model.body_label missing"
        )
        self.shape_is_obstacle = wp.array(is_obstacle, dtype=wp.int32)
        self.contacted = wp.zeros(model.world_count, dtype=wp.int32)
        self.contact_shape = wp.full(model.world_count, -1, dtype=wp.int32)

    def update(self, contacts, body_q):
        wp.launch(
            obstacle_contact_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.shape_is_obstacle,
                self.model.shape_world,
                self.model.shape_body,
                body_q,
            ],
            outputs=[self.contacted, self.contact_shape],
        )

    def read(self):
        return self.contacted.numpy().astype(bool)

    def read_labels(self):
        """Per-world label of the shape that hit the obstacle ("" if none)."""
        s = self.contact_shape.numpy()
        return np.array([self.labels[i] if i >= 0 else "" for i in s])

    def reset(self):
        self.contacted.zero_()
        self.contact_shape.fill_(-1)
