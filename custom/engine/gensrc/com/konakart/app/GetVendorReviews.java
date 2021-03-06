package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - GetVendorReviews - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class GetVendorReviews
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public GetVendorReviews(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public ReviewsIf getVendorReviews(DataDescriptorIf dataDesc, ReviewSearchIf search) throws KKException
     {
         return kkEng.getVendorReviews(dataDesc, search);
     }
}
