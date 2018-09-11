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

import com.konakart.al.CustomerMgr;
import com.konakart.al.KKAppEng;
import com.konakart.app.ReviewSearch;
import com.konakart.appif.StoreIf;
import com.konakart.appif.VendorIf;
import com.konakart.bl.ConfigConstants;

/**
 * Gets called after submitting the write review page.
 */
public class WriteVendorReviewSubmitAction extends BaseAction
{
    private static final long serialVersionUID = 1L;

    private int rating;

    private String reviewText;

    private int vendorId;

    private String vendorStoreId;

    public String execute()
    {
        HttpServletRequest request = ServletActionContext.getRequest();
        HttpServletResponse response = ServletActionContext.getResponse();

        try
        {
            int custId;

            KKAppEng kkAppEng = this.getKKAppEng(request, response);

            custId = this.loggedIn(request, response, kkAppEng, "WriteVendorReview");

            // Go to login page if session has timed out
            if (custId < 0)
            {
                return KKLOGIN;
            }

            if (vendorId <= 0 || vendorStoreId == null)
            {
                return WELCOME;
            }
            CustomerMgr custMgr = kkAppEng.getCustomerMgr();
            if (custMgr.getCurrentVendor() == null
                    || custMgr.getCurrentVendor().getId() != vendorId)
            {
                VendorIf vendor = kkAppEng.getEng().getVendorForId(vendorId);
                if (vendor == null)
                {
                    log.debug("The vendor for id " + vendorId + " does not exist");
                    return WELCOME;
                }
                custMgr.setCurrentVendor(vendor);
            }

            if (custMgr.getCurrentStore() == null
                    || custMgr.getCurrentStore().getStoreId() != vendorStoreId)
            {
                StoreIf store = kkAppEng.getEng().getStoreForId(vendorStoreId);
                if (store == null)
                {
                    log.debug("The store for id " + vendorStoreId + " does not exist");
                    return WELCOME;
                }
                custMgr.setCurrentStore(store);
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

            kkAppEng.getReviewMgr().writeReview(getReviewText(), getRating(), custId,
                    /* productId */0, vendorId);

            // Set reward points if applicable
            if (kkAppEng.getRewardPointMgr().isEnabled())
            {
                String pointsStr = kkAppEng.getConfig(ConfigConstants.REVIEW_REWARD_POINTS);
                if (pointsStr != null)
                {
                    int points = 0;
                    try
                    {
                        points = Integer.parseInt(pointsStr);
                        kkAppEng.getRewardPointMgr().addPoints(points, "REV",
                                kkAppEng.getMsg("reward.points.review"));
                    } catch (Exception e)
                    {
                        log.warn(
                                "The REVIEW_REWARD_POINTS configuration variable has been set with a non numeric value: "
                                        + pointsStr);
                    }
                }
            }
            
            // Re-fetch the vendor with updated statistics
            kkAppEng.getCustomerMgr().setCurrentVendor(kkAppEng.getEng().getVendorForId(vendorId));

            // Get the latest reviews
            ReviewSearch search = new ReviewSearch();
            search.setVendorId(vendorId);
            kkAppEng.getReviewMgr().fetchVendorReviews(null, search, vendorStoreId);
            kkAppEng.getNav().set(kkAppEng.getMsg("header.store"), request);

            return SUCCESS;

        } catch (Exception e)
        {
            return super.handleException(request, e);
        }

    }

    /**
     * @return the rating
     */
    public int getRating()
    {
        return rating;
    }

    /**
     * @param rating
     *            the rating to set
     */
    public void setRating(int rating)
    {
        this.rating = rating;
    }

    /**
     * @return the reviewText
     */
    public String getReviewText()
    {
        return reviewText;
    }

    /**
     * @param reviewText
     *            the reviewText to set
     */
    public void setReviewText(String reviewText)
    {
        this.reviewText = reviewText;
    }

    /**
     * @return the vendorId
     */
    public int getVendorId()
    {
        return vendorId;
    }

    /**
     * @param vendorId
     *            the vendorId to set
     */
    public void setVendorId(int vendorId)
    {
        this.vendorId = vendorId;
    }

    /**
     * @return the vendorStoreId
     */
    public String getVendorStoreId()
    {
        return vendorStoreId;
    }

    /**
     * @param vendorStoreId
     *            the vendorStoreId to set
     */
    public void setVendorStoreId(String vendorStoreId)
    {
        this.vendorStoreId = vendorStoreId;
    }

}
