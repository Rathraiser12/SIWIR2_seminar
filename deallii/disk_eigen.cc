// disk_eigen.cc  — deal.II 9.4  + PETSc + SLEPc  (2‑D unit disk)
// -----------------------------------------------------------------------------
//  −Δu − k²(r) u = λ u   on Ω = {x²+y²<1},  u|Γ = 0
//  k²(r) = (100+δ)·exp(−50 r²) − 100           (δ = 0.01)
// -----------------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_pattern.h>   
#include <deal.II/fe/mapping_q1.h>

#include <fstream>
#include <iomanip>

using namespace dealii;

template <int dim>
class HelmholtzDisk
{
public:
  HelmholtzDisk(unsigned int refine = 4);
  void run();

private:
  void setup_system();
  void assemble_matrices();
  void solve_eigs();
  void write_modes() const;

  MPI_Comm mpi_comm;
  Triangulation<dim>      tria;
  FE_Q<dim>               fe;
  DoFHandler<dim>         dh;
  AffineConstraints<double> constraints;

  PETScWrappers::MPI::SparseMatrix A, M;
  std::vector<double>              eigval;
  std::vector<PETScWrappers::MPI::Vector> eigvec;

  const unsigned int refinements;
  static constexpr double delta = 0.01;
};

/* ------------ constructor ----------------- */
template <int dim>
HelmholtzDisk<dim>::HelmholtzDisk(unsigned int refine)
  : mpi_comm(MPI_COMM_WORLD)
  , fe(1)
  , dh(tria)
  , refinements(refine)
{}

/* ------------ mesh & DoFs ----------------- */
template <int dim>
void HelmholtzDisk<dim>::setup_system()
{
  GridGenerator::hyper_ball(tria, Point<dim>(), 1.0);
  tria.refine_global(refinements);

  dh.distribute_dofs(fe);

  constraints.clear();
  DoFTools::make_zero_boundary_constraints(dh, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dh.n_dofs(), dh.n_dofs());
  DoFTools::make_sparsity_pattern(dh, dsp, constraints, /*keep_constrained*/false);

  SparsityPattern sp;
  sp.copy_from(dsp);

  IndexSet owned = dh.locally_owned_dofs();
  A.reinit(owned, owned, sp, mpi_comm);
  M.reinit(owned, owned, sp, mpi_comm);
}

/* ------------ assemble A & M -------------- */
template <int dim>
void HelmholtzDisk<dim>::assemble_matrices()
{
  QGauss<dim>   quad(fe.degree + 1);
  FEValues<dim> fev(fe, quad,
                    update_values | update_gradients |
                    update_quadrature_points | update_JxW_values);

  const unsigned int ndofs = fe.n_dofs_per_cell();
  const unsigned int nq    = quad.size();

  FullMatrix<double> cellA(ndofs, ndofs), cellM(ndofs, ndofs);
  std::vector<types::global_dof_index> idx(ndofs);

  for (const auto &cell : dh.active_cell_iterators())
  {
    cellA = 0;  cellM = 0;
    fev.reinit(cell);

    for (unsigned q = 0; q < nq; ++q)
    {
      const auto &x = fev.quadrature_point(q);
      const double r2 = x[0]*x[0] + x[1]*x[1];
      const double k2 = (100+delta) * std::exp(-50.*r2) - 100.;

      for (unsigned i = 0; i < ndofs; ++i)
        for (unsigned j = 0; j < ndofs; ++j)
        {
          cellA(i,j) += (fev.shape_grad(i,q)*fev.shape_grad(j,q)
                       -  k2 * fev.shape_value(i,q)*fev.shape_value(j,q))
                       * fev.JxW(q);
          cellM(i,j) += fev.shape_value(i,q)*fev.shape_value(j,q)*fev.JxW(q);
        }
    }
    cell->get_dof_indices(idx);
    constraints.distribute_local_to_global(cellA, idx, A);
    constraints.distribute_local_to_global(cellM, idx, M);
  }

  A.compress(VectorOperation::add);
  M.compress(VectorOperation::add);
}

/* ------------ eigen‑solve ----------------- */
template <int dim>
void HelmholtzDisk<dim>::solve_eigs()
{
  const unsigned int nev = 10;
  eigval.resize(nev);
  eigvec.resize(nev, PETScWrappers::MPI::Vector(dh.locally_owned_dofs(), mpi_comm));

  SolverControl ctrl(1000, 1e-10);
  SLEPcWrappers::SolverKrylovSchur eps(ctrl, mpi_comm);
  eps.set_problem_type(EPS_GHEP);
  eps.set_which_eigenpairs(EPS_SMALLEST_REAL);

  eps.solve(A, M, eigval, eigvec, nev);

  if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::cout << "\nEigenvalues (unit disk, P1, refine=" << refinements << "):\n";
    std::cout << std::setprecision(12);
    for (unsigned i = 0; i < eigval.size(); ++i)
      std::cout << "  λ[" << i << "] = " << eigval[i] << '\n';
  }
}

/* ------------ write mode_k.dat ------------ */
template <int dim>
void HelmholtzDisk<dim>::write_modes() const
{
  MappingQ1<dim> mapping;
  std::vector<Point<dim>> support(dh.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dh, support);

  for (unsigned k = 0; k < eigvec.size(); ++k)
  {
    std::ofstream out("mode_" + Utilities::to_string(k) + ".dat");
    for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
      out << support[i][0] << ' ' << support[i][1] << ' ' << eigvec[k][i] << '\n';
  }
}

/* ------------ driver ---------------------- */
template <int dim>
void HelmholtzDisk<dim>::run()
{
  setup_system();
  assemble_matrices();
  solve_eigs();
  write_modes();
}

/* ------------ main ------------------------ */
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  HelmholtzDisk<2> problem(4);   // 4 global refinements ≈  h ≈ 1/16
  problem.run();
}
