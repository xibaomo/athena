/*
 * =====================================================================================
 *
 *       Filename:  mtqueue.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 08:18:35 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BASICS_MTQUEUE_H_
#define  _BASICS_MTQUEUE_H_

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <queue>
template <typename T>
class MtQueue {
private:
    std::queue<T> m_queue;
    mutable boost::mutex m_mutex;
    boost::condition_variable m_condvar;
public:
    void push(T const& data) {
        boost::mutex::scoped_lock lock(m_mutex);
        m_queue.push(data);
        lock.unlock();
        m_condvar.notify_one();
    }

    void push(T&& data) {
        boost::mutex::scoped_lock lock(m_mutex);
        m_queue.push(std::move(data));
        lock.unlock();
        m_condvar.notify_one();
    }

    bool empty() const {
        boost::mutex::scoped_lock lock(m_mutex);
        return m_queue.empty();
    }

    size_t size() const {
        boost::mutex::scoped_lock lock(m_mutex);
        return m_queue.size();
    }

    T& front() {
        boost::mutex::scoped_lock lock(m_mutex);
        return m_queue.front();
    }

    void pop() {
        boost::mutex::scoped_lock lock(m_mutex);
        m_queue.pop();
    }
};
#endif   /* ----- #ifndef _BASICS_MTQUEUE_H_  ----- */
