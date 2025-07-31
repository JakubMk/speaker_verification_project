import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf

@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    # dataset
    classes, train_ds, validation_ds = instantiate(cfg.dataset)

    # normalization and cosine layer
    norm_layer = instantiate(cfg.verification_model.model.normalization_layer)
    cosine_layer_partial = instantiate(cfg.verification_model.model.cosine_layer)
    cosine_layer = cosine_layer_partial(out_features=len(classes))

    # build model
    if not cfg.verification_model.finetune:
        def build_resnet_model():
            ResNet, _ = instantiate(cfg.verification_model.resnet)

            base_cnn = ResNet(input_shape=(512, None, 3), include_top=False, weights=None)
            gap_layer = instantiate(cfg.verification_model.global_pool)
            x = gap_layer(base_cnn.output)
            
            return tf.keras.Model(inputs=base_cnn.input, outputs=x, name=cfg.verification_model.resnet_name)

        base_model = build_resnet_model()

        # instantiate the verification model
        model_partial = instantiate(cfg.verification_model.model)
        model = model_partial(base_model=base_model,
                              number_of_classes=len(classes),
                              normalization_layer=norm_layer,
                              cosine_layer=cosine_layer,
                              embedding_dim=cfg.verification_model.model.embedding_dim,
                              return_embedding=cfg.verification_model.model.return_embedding,
                              name=cfg.verification_model.model.name)
    else:
        model = instantiate(cfg.verification_model.load_model)

    epoch_init = cfg.stage1.epochs
    def scheduler(epoch, lr, epoch_init=epoch_init):
        if epoch == epoch_init:
            return lr * 0.1
        else:
            return lr * 0.2


    # callbacks
    checkpoint_cb = instantiate(cfg.callbacks.checkpoint)
    reduce_lr_cb = instantiate(cfg.callbacks.reduce_lr_on_plateau)
    lr_scheduler_partial = instantiate(cfg.callbacks.lr_scheduler_partial)
    lr_scheduler = lr_scheduler_partial(schedule=scheduler, verbose=1) 
    eer_cb = instantiate(cfg.callbacks.eer_monitor)
    tensorboard_callback = instantiate(cfg.callbacks.tensorboard_callback)

    loss_partial = hydra.utils.instantiate(cfg.stage1.loss_fn_partial)


    # model compilation
    model.compile(
        optimizer=instantiate(cfg.stage1.optimizer),
        loss=loss_partial(num_classes=len(classes)),
        metrics=[cfg.stage1.metrics]
    )

    print(model.summary())
    
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        initial_epoch=cfg.stage1.initial_epoch,
        epochs=cfg.stage1.epochs,
        callbacks=[eer_cb, checkpoint_cb, reduce_lr_cb, tensorboard_callback])
    
    if cfg.stage2.resume_best:
        model = tf.keras.models.load_model(eer_cb.model_path)

    model.compile(
        optimizer=instantiate(cfg.stage2.optimizer),
        loss=loss_partial(num_classes=len(classes)),
        metrics=[cfg.stage2.metrics],
    )


    for layer in model.base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    history2 = model.fit(
        train_ds,
        validation_data=validation_ds,
        initial_epoch=cfg.stage2.initial_epoch,
        epochs=cfg.stage2.epochs,
        callbacks=[eer_cb, checkpoint_cb, lr_scheduler, tensorboard_callback],
    )

if __name__ == "__main__":
    main()
