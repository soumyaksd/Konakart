//
// (c) 2017 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//

package com.konakart.actions;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.struts2.ServletActionContext;

import com.konakart.al.KKAppEng;
import com.konakart.app.ReviewSearch;
import com.konakart.appif.StoreIf;
import com.konakart.appif.VendorIf;

/**
 * Called to show the details of a store.
 */
public class StoreDetailsAction extends BaseAction
{
    private static final long serialVersionUID = 1L;

    private String id;

    public String execute()
    {
        HttpServletRequest request = ServletActionContext.getRequest();
        HttpServletResponse response = ServletActionContext.getResponse();

        try
        {

            KKAppEng kkAppEng = this.getKKAppEng(request, response);

            int custId = this.loggedIn(request, response, kkAppEng, null);

            // Force the user to login if configured to do so
            if (custId < 0 && kkAppEng.isForceLogin())
            {
                return KKLOGIN;
            }

            // Ensure we are using the correct protocol. Redirect if not.
            String redirForward = checkSSL(kkAppEng, request, custId, /* forceSSL */false);
            if (redirForward != null)
            {
                setupResponseForSSLRedirect(response, redirForward);
                return null;
            }

            if (id == null || id.length() == 0)
            {
                return WELCOME;
            }

            StoreIf store = kkAppEng.getEng().getStoreForId(id);
            kkAppEng.getCustomerMgr().setCurrentStore(store);
            if (store == null)
            {
                log.debug("The store for id " + id + " does not exist");
                return WELCOME;
            }

            VendorIf vendor = null;
            if (store.getVendorId() > 0)
            {
                vendor = kkAppEng.getEng().getVendorForId(store.getVendorId());
                // Get the vendor reviews
                if (vendor != null)
                {
                    ReviewSearch search = new ReviewSearch();
                    search.setVendorId(vendor.getId());
                    kkAppEng.getReviewMgr().fetchVendorReviews(null, search, store.getStoreId());
                }
            }
            kkAppEng.getCustomerMgr().setCurrentVendor(vendor);

            kkAppEng.getNav().set(kkAppEng.getMsg("header.store"), request);
            return SUCCESS;

        } catch (Exception e)
        {
            return super.handleException(request, e);
        }
    }

    /**
     * @return the id
     */
    public String getId()
    {
        return id;
    }

    /**
     * @param id
     *            the id to set
     */
    public void setId(String id)
    {
        this.id = id;
    }
}