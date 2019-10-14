/*
 * =====================================================================================
 *
 *       Filename:  poly_container.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/12/2019 12:02:19 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _POLY_CONTAINER_H_
#define  _POLY_CONTAINER_H_

#include "generic_poly.h"
class PolyContainer {
protected:
    PolySet* m_polys;
    PolyContainer() {;}
public:
    virtual void convertPolys(const String& layer) = 0;

    void loadPolys(PolySet* ps) { m_polys = ps; }
};
#endif   /* ----- #ifndef _POLY_CONTAINER_H_  ----- */
