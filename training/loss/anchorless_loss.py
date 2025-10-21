import tensorflow as tf

def _split_nc(y):                   # y: [N,3]
    return y[..., 0:1], y[..., 1:3]  # (hm, offs)

def _pos_mask_from_hm_nc(hm_true, thresh=0.999):
    return tf.cast(hm_true > thresh, tf.float32)  # [N,1]

def _mean_over_nodes(numer_map, denom_map, eps=1.0):
    numer = tf.reduce_sum(numer_map, axis=[0, 1])                  # scalar
    denom = tf.maximum(tf.reduce_sum(denom_map, axis=[0, 1]), eps) # scalar
    return numer / denom

class AnchorlessLoss:
    """
    Unbatched CenterNet loss.
      - Heatmap: focal loss from logits (1 ch)
      - Offsets: masked MSE over positives (2 ch)
    Call signature:
      __call__(y_true, y_pred, off_mask=None) -> scalar
    If off_mask is None, we infer it from hm_true via a high threshold.
    """
    def __init__(self, alpha=2.0, beta=4.0,
                 offset_weight=1.0,
                 hm_peak_thresh=0.999,
                 reg_pos_thresh=0.999):
        self.alpha = alpha
        self.beta = beta
        self.offset_weight = offset_weight
        self.hm_peak_thresh = hm_peak_thresh
        self.reg_pos_thresh = reg_pos_thresh

    def _focal_heatmap_loss(self, hm_true, hm_logits):
        x   = tf.clip_by_value(hm_logits, -50.0, 50.0)
        p   = tf.sigmoid(x)
        lsp = tf.clip_by_value(tf.math.log_sigmoid(x),  -50.0, 0.0)
        lsn = tf.clip_by_value(tf.math.log_sigmoid(-x), -50.0, 0.0)

        pos   = hm_true
        neg   = 1.0 - pos
        neg_w = tf.pow(1.0 - hm_true, self.beta)

        pos_loss = - tf.pow(1.0 - p, self.alpha) * lsp * pos
        neg_loss = - tf.pow(p,         self.alpha) * lsn * neg * neg_w
        loss_map = pos_loss + neg_loss                         # [N,1]

        obj_map = _pos_mask_from_hm_nc(hm_true, self.hm_peak_thresh)  # [N,1]
        return _mean_over_nodes(loss_map, obj_map, eps=1.0)

    def _mse_offset_loss(self, off_true, off_pred, off_mask):
        # off_true/off_pred: [N,2], off_mask: [N,1] in {0,1}
        sq_err = tf.reduce_sum(tf.square(off_pred - off_true), axis=-1, keepdims=True)  # [N,1]
        return _mean_over_nodes(sq_err * off_mask, off_mask, eps=1.0)

    def __call__(self, y_true, y_pred, off_mask=None):
        # y_true/y_pred: [N,3]
        y_true = tf.ensure_shape(y_true, [None, 3])
        y_pred = tf.ensure_shape(y_pred, [None, 3])

        hm_t, off_t      = _split_nc(y_true)   # [N,1], [N,2]
        hm_logits, off_p = _split_nc(y_pred)   # [N,1], [N,2]

        # If no explicit mask was provided, derive it from hard peaks in hm_true
        if off_mask is None:
            off_mask = _pos_mask_from_hm_nc(hm_t, self.reg_pos_thresh)  # [N,1]
        else:
            # Ensure float dtype & correct rank
            off_mask = tf.cast(off_mask, tf.float32)
            off_mask = tf.ensure_shape(off_mask, [None, 1])

        hm_loss  = self._focal_heatmap_loss(hm_t, hm_logits)
        off_loss = self._mse_offset_loss(off_t, off_p, off_mask)
        return hm_loss + self.offset_weight * off_loss
