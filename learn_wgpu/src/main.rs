use std::sync::Arc;

use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{Window}
};

#[cfg(target_arch ="wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

//store state of game
pub struct State {
    window: Arc<Window>,
}

impl State{
    pub async fn new(window:Arc<Window>) -> anyhow::Result<Self> {
        Ok(Self { 
            window,
        })
    }

    pub fn resize(&mut self, _width:u32, _height:u32) {

    }

    pub fn render(&mut self) {
        self.window.request_redraw();
    }
}