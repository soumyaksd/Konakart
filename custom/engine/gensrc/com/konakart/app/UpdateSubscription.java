package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - UpdateSubscription - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class UpdateSubscription
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public UpdateSubscription(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public void updateSubscription(String sessionId, SubscriptionIf subscription) throws KKException
     {
         kkEng.updateSubscription(sessionId, subscription);
     }
}