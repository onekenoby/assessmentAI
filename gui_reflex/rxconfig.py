import reflex as rx

config = rx.Config(
    app_name="gui_reflex",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        # Ecco il tuo tema spostato qui:
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="light", 
                accent_color="indigo", 
                radius="large"
            )
        ),
    ]
)