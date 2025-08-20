The internal API
================

.. automodule:: plscan.api

   .. rubric:: Classes

   .. autosummary::
      :toctree: reference_api

      NodeData
      SpaceTree
      SparseGraph
      SpanningTree
      LinkageTree
      CondensedTree
      LeafTree
      PersistenceTrace
      Labelling

   .. rubric:: Threads

   .. autosummary::
      :toctree: reference_api

      get_max_threads
      set_num_threads

   .. rubric:: Space tree

   .. autosummary::
      :toctree: reference_api

      kdtree_query
      balltree_query

   .. rubric:: Sparse graph

   .. autosummary::
      :toctree: reference_api

      extract_core_distances
      compute_mutual_reachability

   .. rubric:: Spanning tree

   .. autosummary::
      :toctree: reference_api

      extract_spanning_forest
      compute_spanning_tree_kdtree
      compute_spanning_tree_balltree

   .. rubric:: Linkage tree

   .. autosummary::
      :toctree: reference_api

      compute_linkage_tree

   .. rubric:: Condense tree

   .. autosummary::
      :toctree: reference_api

      compute_condensed_tree

   .. rubric:: Leaf tree

   .. autosummary::
      :toctree: reference_api

      compute_leaf_tree
      apply_size_cut
      apply_distance_cut

   .. rubric:: Persistence trace

   .. autosummary::
      :toctree: reference_api
      
      compute_bi_persistence
      compute_size_persistence
      compute_stability_icicles

   .. rubric:: Labelling

   .. autosummary::
      :toctree: reference_api

      compute_cluster_labels

   .. rubric:: Sklearn helpers

   .. autosummary::
      :toctree: reference_api

      distance_matrix_to_csr
      knn_to_csr
      remove_self_loops
      sort_spanning_tree
      most_persistent_clusters