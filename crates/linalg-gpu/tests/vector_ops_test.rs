// crates/linalg-gpu/tests/vector_ops_test.rs
use fem_linalg_gpu::{GpuContext, GpuVector, VectorOpsPipeline};

fn ctx() -> GpuContext {
    GpuContext::new_sync().expect("gpu context")
}

#[test]
fn axpy_simple() {
    let gpu = ctx();
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);

    let x = GpuVector::from_slice(&gpu, &[1.0f64, 2.0, 3.0]);
    let y = GpuVector::from_slice(&gpu, &[4.0f64, 5.0, 6.0]);

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    // y = 2*x + 0.5*y = [2+2, 4+2.5, 6+3] = [4, 6.5, 9]
    pipeline.encode_axpy(&gpu, &mut encoder, 2.0, &x, 0.5, &y);
    gpu.queue.submit(Some(encoder.finish()));

    let result = y.read_to_cpu(&gpu);
    assert!((result[0] - 4.0).abs() < 1e-14);
    assert!((result[1] - 6.5).abs() < 1e-14);
    assert!((result[2] - 9.0).abs() < 1e-14);
}

#[test]
fn dot_simple() {
    let gpu = ctx();
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);

    let a = GpuVector::from_slice(&gpu, &[1.0f64, 2.0, 3.0]);
    let b = GpuVector::from_slice(&gpu, &[4.0f64, 5.0, 6.0]);
    let n_wg = (a.len() + 255) / 256;
    let result_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dot_result"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    pipeline.encode_dot(&gpu, &mut encoder, &a, &b, &result_buf);
    gpu.queue.submit(Some(encoder.finish()));

    // Read back the partial reduction result and sum on CPU
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dot_staging"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&result_buf, 0, &staging, 0, n_wg as u64 * 8);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let partials: &[f64] = bytemuck::cast_slice(&mapped);
    let dot: f64 = partials.iter().sum();
    drop(mapped);
    let _ = slice;
    staging.unmap();

    let expected = 1.0*4.0 + 2.0*5.0 + 3.0*6.0; // = 4+10+18 = 32
    assert!((dot - expected).abs() < 1e-13, "dot={dot} expected={expected}");
}
