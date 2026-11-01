

#include "cpu/relabel2hop_cpu.cpp"

TORCH_LIBRARY_FRAGMENT(torch_geometric_autoscale, m) {
  m.class_<Relabel2hop<std::string>>("Relabel2hop")
    .def(torch::init<torch::Tensor,torch::Tensor,torch::optional<torch::Tensor>>())
    .def("relabel_one_hop", &Relabel2hop<std::string>::relabel_one_hop)
    .def("relabel_one_hop_compensation", &Relabel2hop<std::string>::relabel_one_hop_compensation)
    .def("relabel_one_hop_compensation_rm2hop", &Relabel2hop<std::string>::relabel_one_hop_compensation_rm2hop);
}

 