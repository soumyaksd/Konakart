//
// (c) 2006-2017 DS Data Systems UK Ltd, All rights reserved.
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

import com.konakart.al.CustomerMgr;
import com.konakart.al.KKAppEng;
import com.konakart.appif.StoreIf;
import com.konakart.appif.VendorIf;

/**
 * Gets called before the write review page.
 */
public class WriteVendorReviewAction extends BaseAction
{
    private static final long serialVersionUID = 1L;

    public String execute()
    {
        HttpServletRequest request = ServletActionContext.getRequest();
        HttpServletResponse response = ServletActionContext.getResponse();

        try
        {

            KKAppEng kkAppEng = this.getKKAppEng(request, response);

            // Check to see whether the user is logged in
            int custId = this.loggedIn(request, response, kkAppEng, "WriteVendorReview");
            if (custId < 0)
            {
                return KKLOGIN;
            }

            String vendorId = request.getParameter("vid");
            String vendorStoreId = request.getParameter("sid");

            // vendorId may be set to null if we are forced to login and then sent back to this
            // action. However, in this case the selected vendor has already been set, so we use
            // that.

            CustomerMgr custMgr = kkAppEng.getCustomerMgr();

            if (vendorId == null)
            {
                if (custMgr.getCurrentVendor() == null)
                {
                    log.debug(
                            "The vendor Id for the review cannot be set to null because the currentVendor is also set to null");
                    return WELCOME;
                }
                // At this point we use the current vendor in the custMgr
            } else
            {
                int vendorIdInt;
                try
                {
                    vendorIdInt = Integer.parseInt(vendorId);
                } catch (Exception e)
                {
                    return WELCOME;
                }

                if (custMgr.getCurrentVendor() == null
                        || custMgr.getCurrentVendor().getId() != vendorIdInt)
                {
                    VendorIf vendor = kkAppEng.getEng().getVendorForId(vendorIdInt);
                    if (vendor == null)
                    {
                        log.debug("The vendor for id " + vendorId + " does not exist");
                        return WELCOME;
                    }
                    custMgr.setCurrentVendor(vendor);
                }
            }

            if (vendorStoreId == null)
            {
                if (custMgr.getCurrentStore() == null)
                {
                    log.debug(
                            "The vendor store id cannot be set to null because the currentStore is also set to null");
                    return WELCOME;
                }
            } else
            {
                if (custMgr.getCurrentStore() == null
                        || custMgr.getCurrentStore().getStoreId() != vendorStoreId)
                {
                    StoreIf store = kkAppEng.getEng().getStoreForId(vendorStoreId);
                    if (store == null)
                    {
                        log.debug("The store for id " + vendorStoreId + " does not exist");
                        return WELCOME;
                    }
                    kkAppEng.getCustomerMgr().setCurrentStore(store);
                }
            }
            
            if (custMgr.getCurrentStore().getVendorId() != custMgr.getCurrentVendor().getId())
            {
                log.debug("The vendor id of the store " + custMgr.getCurrentStore().getVendorId()
                        + " does not match the vendor id " + custMgr.getCurrentVendor().getId());
                return WELCOME;
            }

            // Ensure we are using the correct protocol. Redirect if not.
            String redirForward = checkSSL(kkAppEng, request, custId, /* forceSSL */false);
            if (redirForward != null)
            {
                setupResponseForSSLRedirect(response, redirForward);
                return null;
            }

            kkAppEng.getNav().set(kkAppEng.getMsg("header.write.review"), request);

            return SUCCESS;

        } catch (Exception e)
        {
            return super.handleException(request, e);
        }
    }
}
