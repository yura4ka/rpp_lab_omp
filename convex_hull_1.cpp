#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

struct Point
{
    double x, y;

    Point(double x_val, double y_val) : x(x_val), y(y_val) {}

    double angle(const Point& p1) const
    {
        return std::atan2(p1.y - this->y, p1.x - this->x);
    }

    double distance(const Point& p1) const
    {
        return std::pow(p1.x - this->x, 2) + std::pow(p1.y - this->y, 2);
    }
};

int orientation(const Point& p1, const Point& p2, const Point& p3)
{
    double val = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    if (val == 0) return 0;
    if (val > 0) return 1;
    return -1;
}

std::vector<Point> convexHull(std::vector<Point>& points) {
    if (points.size() <= 3) return points;

    int id = omp_get_thread_num();
    int num_of_threads = omp_get_num_threads();
    int i_start = id * points.size() / num_of_threads;
    int i_end = (id + 1) * points.size() / num_of_threads;
    if (id == num_of_threads - 1)
    {
        i_end = points.size();
    }

    Point start = *std::min_element(points.begin() + i_start, points.begin() + i_end, [](const Point& p1, const Point& p2) {
        return p1.y < p2.y || (p1.y == p2.y && p1.x < p2.x);
        });

    std::sort(points.begin() + i_start, points.begin() + i_end, [&start](const Point& p1, const Point& p2) {
        return start.angle(p1) < start.angle(p2) || (start.angle(p1) == start.angle(p2) && start.distance(p1) < start.distance(p2));
        });

    std::vector<Point> hull = { points[i_start], points[i_start + 1] };

    for (int i = i_start + 2; i < i_end; ++i)
    {
        while (hull.size() > 1 && orientation(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    return hull;
}

std::vector<Point> input_points(const char* filename)
{
    int n;
    std::vector<Point> points;

    std::ifstream file(filename);
    file >> n;

    for (int i = 0; i < n; ++i)
    {
        double x, y;
        file >> x >> y;
        points.push_back(Point(x, y));
    }

    return points;
}

int main()
{
    auto points = input_points("1000000_1.txt");
    std::vector<Point> result;
    double time_start = omp_get_wtime(), time_end;

#pragma omp parallel 
    {
        auto partial = convexHull(points);
#pragma omp critical 
        {
            result.insert(result.end(), partial.begin(), partial.end());
        }
    }

    auto hull = convexHull(result);
    time_end = omp_get_wtime();
    std::cout << "Input size: " << points.size()
        << "\nNumber of threads: " << omp_get_num_threads()
        << "\nTime: " << time_end - time_start
        << "\nHull size: " << hull.size() << "\n";
    return 0;
}
