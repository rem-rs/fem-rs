// crates/linalg-gpu/tests/vector_ops_test.rs
use fem_core::Scalar;
use fem_linalg_gpu::{GpuContext, GpuVector, VectorOpsPipeline};

fn ctx() -> GpuContext {
    GpuContext::new_sync().expect("gpu context")
}

fn assert_close<T: Scalar>(actual: T, expected: T, tol: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(diff < T::from_f64(tol), "{label}: actual={actual} expected={expected} diff={diff}");
}

fn run_axpy_simple<T: Scalar>(gpu: &GpuContext, tol: f64) {
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);

    let x = GpuVector::from_slice(gpu, &[T::from_f64(1.0), T::from_f64(2.0), T::from_f64(3.0)]);
    let y = GpuVector::from_slice(gpu, &[T::from_f64(4.0), T::from_f64(5.0), T::from_f64(6.0)]);

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    pipeline.encode_axpy(gpu, &mut encoder, 2.0, &x, 0.5, &y);
    gpu.queue.submit(Some(encoder.finish()));

    let result = y.read_to_cpu(gpu);
    assert_close(result[0], T::from_f64(4.0), tol, "axpy[0]");
    assert_close(result[1], T::from_f64(6.5), tol, "axpy[1]");
    assert_close(result[2], T::from_f64(9.0), tol, "axpy[2]");
}

fn run_dot_simple<T: Scalar>(gpu: &GpuContext, tol: f64) {
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);

    let a = GpuVector::from_slice(gpu, &[T::from_f64(1.0), T::from_f64(2.0), T::from_f64(3.0)]);
    let b = GpuVector::from_slice(gpu, &[T::from_f64(4.0), T::from_f64(5.0), T::from_f64(6.0)]);
    let n_wg = (a.len() + 255) / 256;
    let elem_size = std::mem::size_of::<T>() as u64;
    let result_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dot_result"),
        size: n_wg as u64 * elem_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    pipeline.encode_dot(gpu, &mut encoder, &a, &b, &result_buf);
    gpu.queue.submit(Some(encoder.finish()));

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dot_staging"),
        size: n_wg as u64 * elem_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&result_buf, 0, &staging, 0, n_wg as u64 * elem_size);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let partials: &[T] = bytemuck::cast_slice(&mapped);
    let dot = partials.iter().copied().fold(T::zero(), |acc, value| acc + value);
    drop(mapped);
    staging.unmap();

    assert_close(dot, T::from_f64(32.0), tol, "dot");
}

#[test]
fn axpy_simple() {
    let gpu = ctx();
    if gpu.features.native_f64 {
        run_axpy_simple::<f64>(&gpu, 1e-14);
    } else {
        run_axpy_simple::<f32>(&gpu, 1e-5);
    }
}

#[test]
fn dot_simple() {
    let gpu = ctx();
    if gpu.features.native_f64 {
        run_dot_simple::<f64>(&gpu, 1e-13);
    } else {
        run_dot_simple::<f32>(&gpu, 1e-4);
    }
}
