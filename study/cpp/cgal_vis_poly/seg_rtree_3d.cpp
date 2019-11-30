#include <CGAL/Cartesian.h>
#include <CGAL/Segment_tree_k.h>
#include <CGAL/Range_segment_tree_traits.h>
#include "utils.h"
#include <iostream>
using namespace std;
typedef CGAL::Cartesian<int> K;
typedef CGAL::Range_segment_tree_set_traits_3<K> Traits;
typedef CGAL::Segment_tree_3<Traits> Segment_tree_3_type;
int main()
{
  typedef Traits::Interval Interval;
  typedef Traits::Key Key;
  std::list<Interval> InputList, OutputList;
  InputList.push_back(Interval(Key(1, 5, 7), Key(2, 7, 9)));
  InputList.push_back(Interval(Key(2, 7, 6), Key(3, 8, 9)));
  InputList.push_back(Interval(Key(6, 9, 5), Key(9, 13, 8)));
  InputList.push_back(Interval(Key(1, 3, 4), Key(3, 9, 8)));
  Segment_tree_3_type Segment_tree_3(InputList.begin(), InputList.end());
  Interval a(Key(3, 6, 5), Key(7, 12, 8));
  MsTimer tim;
  Segment_tree_3.window_query(a, std::back_inserter(OutputList));
  cout << "query takes " << tim.elapsed() << endl;
  std::list<Interval>::iterator j = OutputList.begin();
  std::cout << "\n window_query (3, 6, 5), (7, 12, 8) \n";
  while ( j!=OutputList.end() ) {
    std::cout << (*j).first.x() << "," << (*j).first.y() << ",";
    std::cout << (*j).first.z() << ", " << (*j).second.x() << ",";
    std::cout << (*j).second.y() << "," << (*j).second.z() << std::endl;
    j++;
  }
  return 0;
}
