#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkflowCliOptions {
    pub checkpoint: Option<String>,
    pub checkpoint_h5: Option<String>,
    pub restart: Option<String>,
    pub restart_h5: Option<String>,
    pub export_vtk_prefix: Option<String>,
}

impl WorkflowCliOptions {
    pub fn try_assign(&mut self, arg: &str, value: Option<String>) -> bool {
        match arg {
            "--checkpoint" => {
                self.checkpoint = value;
                true
            }
            "--checkpoint-h5" => {
                self.checkpoint_h5 = value;
                true
            }
            "--restart" => {
                self.restart = value;
                true
            }
            "--restart-h5" => {
                self.restart_h5 = value;
                true
            }
            "--export-vtk-prefix" => {
                self.export_vtk_prefix = value;
                true
            }
            _ => false,
        }
    }

    pub fn try_parse_arg<I>(&mut self, arg: &str, it: &mut I) -> bool
    where
        I: Iterator<Item = String>,
    {
        self.try_assign(arg, it.next())
    }
}

pub fn assert_single_restart_source(options: &WorkflowCliOptions) {
    assert!(
        options.restart.is_none() || options.restart_h5.is_none(),
        "choose only one restart source: --restart or --restart-h5"
    );
}

pub fn push_workflow_cli_help(
    options: &mut Vec<(&'static str, &'static str)>,
    export_vtk_desc: &'static str,
) {
    options.extend([
        ("--checkpoint <path>", "Write lightweight text checkpoint at end of run"),
        ("--checkpoint-h5 <path>", "Write shared HDF5 checkpoint at end of run"),
        ("--restart <path>", "Restart from a lightweight text checkpoint"),
        ("--restart-h5 <path>", "Restart from a shared HDF5 checkpoint"),
        ("--export-vtk-prefix <prefix>", export_vtk_desc),
    ]);
}