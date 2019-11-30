#include <CGAL/Cartesian.h>
#include <CGAL/Segment_tree_k.h>
#include <CGAL/Range_segment_tree_traits.h>
#include "utils.h"
#include <iostream>
#include <vector>
using namespace std;
typedef CGAL::Cartesian<double> K;
typedef CGAL::Range_segment_tree_set_traits_2<K> Traits;
typedef CGAL::Segment_tree_2<Traits> Segment_tree_2_type;
int main()
{
    const int N = 200;
    typedef Traits::Interval Interval;
    typedef Traits::Key Key;
    std::list<Interval> InputList, OutputList;
    for ( int i = 0; i < N; i++ ) {
        InputList.push_back(Interval(Key(i, i), Key(i+0.5, i+0.5)));
    }

    MsTimer tim;
    Segment_tree_2_type Segment_tree_2(InputList.begin(), InputList.end());
    cout << "build takes " << tim.elapsed() << endl;
    Interval a(Key(3, 6), Key(7, 12));
    tim.restart();
    Segment_tree_2.window_query(a, std::back_inserter(OutputList));
    cout << "query takes " << tim.elapsed() << endl;
    auto j = OutputList.begin();
    std::cout << "\n window_query (3, 6), (7, 12) \n";
    while ( j!=OutputList.end() ) {
        std::cout << (*j).first.x() << "," << (*j).first.y() << ",";
        std::cout << ", " << (*j).second.x() << ",";
        std::cout << (*j).second.y() << "," << std::endl;
        j++;
    }
    return 0;
}
