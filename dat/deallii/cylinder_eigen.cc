// disk_eigen.cc  -- deal.II 9.4  + PETSc + SLEPc
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <fstream>
#include <iomanip>

using namespace dealii;

/* ------------------------------------------------------------------ */
template <int dim>
void write_mode_dat(const DoFHandler<dim>            &dh,
                    const PETScWrappers::MPI::Vector &vec,
                    const std::string                &fname)
{
  MappingQ1<dim> mapping;
  std::vector<Point<dim>> support(dh.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dh, support);

  std::ofstream out(fname);
  out << "# x  y  z  value\n";
  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
  {
    const auto &p = support[i];
    out << p[0] << ' ' << p[1] << ' '
        << (dim == 3 ? p[2] : 0.0) << ' '
        << vec[i] << '\n';
  }
}
/* ------------------------------------------------------------------ */

template <int dim>
class HelmholtzDisk
{
public:
  HelmholtzDisk(unsigned int refinements = 2);
  void run();

private:
  void setup_system();
  void assemble();
  void solve();
  void output_modes() const;

  MPI_Comm mpi_comm;
  Triangulation<dim> tria;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dh;

  PETScWrappers::MPI::SparseMatrix A, M;
  std::vector<double>              eigval;
  std::vector<PETScWrappers::MPI::Vector> eigvec;

  const unsigned int refinements;
  static constexpr double delta = 0.01;
};

/* ---- ctor -------------------------------------------------------- */
template <int dim>
HelmholtzDisk<dim>::HelmholtzDisk(unsigned int ref)
  : mpi_comm(MPI_COMM_WORLD)
  , fe(1)
  , dh(tria)
  , refinements(ref)
{}

/* ---- system setup ------------------------------------------------ */
template <int dim>
void HelmholtzDisk<dim>::setup_system()
{
  GridGenerator::hyper_ball(tria, Point<dim>(), 1.0);
  tria.refine_global(refinements);
  dh.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dh.n_dofs(), dh.n_dofs());
  DoFTools::make_sparsity_pattern(dh, dsp);

  SparsityPattern sp;
  sp.copy_from(dsp);

  IndexSet owned = dh.locally_owned_dofs();
  A.reinit(owned, owned, sp, mpi_comm);
  M.reinit(owned, owned, sp, mpi_comm);
}

/* ---- assemble A and M ------------------------------------------- */
template <int dim>
void HelmholtzDisk<dim>::assemble()
{
  QGauss<dim>   quad(fe.degree + 1);
  FEValues<dim> fev(fe, quad,
                    update_values | update_gradients |
                    update_quadrature_points | update_JxW_values);

  const unsigned int dofs = fe.n_dofs_per_cell();
  const unsigned int nq   = quad.size();

  FullMatrix<double> cellA(dofs, dofs), cellM(dofs, dofs);
  std::vector<types::global_dof_index> idx(dofs);

  for (const auto &cell : dh.active_cell_iterators())
  {
    cellA = 0; cellM = 0;
    fev.reinit(cell);

    for (unsigned int q = 0; q < nq; ++q)
    {
      const auto &x = fev.quadrature_point(q);
      const double r2 = x[0]*x[0] + x[1]*x[1];
      const double k2 = (100 + delta) * std::exp(-50. * r2) - 100.;

      for (unsigned int i = 0; i < dofs; ++i)
        for (unsigned int j = 0; j < dofs; ++j)
        {
          cellA(i,j) += (fev.shape_grad(i,q)*fev.shape_grad(j,q)
                       -  k2 * fev.shape_value(i,q)*fev.shape_value(j,q))
                       * fev.JxW(q);
          cellM(i,j) += fev.shape_value(i,q)*fev.shape_value(j,q)*fev.JxW(q);
        }
    }
    cell->get_dof_indices(idx);
    A.add(idx, cellA);
    M.add(idx, cellM);
  }
  A.compress(VectorOperation::add);
  M.compress(VectorOperation::add);
}

/* ---- solve with SLEPc ------------------------------------------- */
template <int dim>
void HelmholtzDisk<dim>::solve()
{
  const unsigned int nev = 10;
  eigval.resize(nev);
  eigvec.resize(nev, PETScWrappers::MPI::Vector(dh.locally_owned_dofs(),
                                                mpi_comm));

  SolverControl control(1000, 1e-8);
  SLEPcWrappers::SolverKrylovSchur eps(control, mpi_comm);
  eps.set_problem_type(EPS_GHEP);
  eps.set_which_eigenpairs(EPS_SMALLEST_REAL);

  eps.solve(A, M, eigval, eigvec, nev);

  if (Utilities::MPI::this_mpi_process(mpi_comm)==0)
  {
    std::cout << "\nEigenvalues:\n";
    for (unsigned int i=0;i<nev && i<eigval.size();++i)
      std::cout << " λ["<<i<<"] = "
                << std::setprecision(12) << eigval[i] << '\n';
    std::cout << '\n';
  }
}

/* ---- write mode_k.dat ------------------------------------------ */
template <int dim>
void HelmholtzDisk<dim>::output_modes() const
{
  for (unsigned int k=0;k<eigvec.size();++k)
  {
    const std::string fname = "mode_" + Utilities::to_string(k) + ".dat";
    write_mode_dat(dh, eigvec[k], fname);
  }
  if (Utilities::MPI::this_mpi_process(mpi_comm)==0)
    std::cout << "Wrote " << eigvec.size() << " files  mode_*.dat\n";
}

/* ---- driver ----------------------------------------------------- */
template <int dim>
void HelmholtzDisk<dim>::run()
{
  setup_system();
  assemble();
  solve();
  output_modes();
}

/* ---- main ------------------------------------------------------- */
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  HelmholtzDisk<2> problem(4);    // You can vary the number of refinements
  problem.run();
}
