/******************************************************************************
 *                                                                            *
 * Copyright (C) 2019 Fondazione Istituto Italiano di Tecnologia (IIT)        *
 * All Rights Reserved.                                                       *
 *                                                                            *
 ******************************************************************************/

/**
 * @file main.cpp
 * @authors: Ugo Pattacini <ugo.pattacini@iit.it>
 */

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include "AssistiveRehab/skeleton.h"

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;
using namespace assistive_rehab;

const string unknown_tag("?");

/****************************************************************/
bool is_unknown(const string &tag)
{
    return (tag.empty() || (tag==unknown_tag));
}

/****************************************************************/
class Locker : public RFModule
{
    BufferedPort<Bottle> skeletonsPort;
    BufferedPort<Bottle> viewerPort;
    RpcClient opcPort;

    vector<shared_ptr<MetaSkeleton>> skeletons;

    double period{0.01};
    double tracking_threshold{0.5};
    unordered_map<string,string> keypoints_map;
    unordered_set<string> tracking_keypoints{KeyPointTag::shoulder_center,KeyPointTag::hip_center};
    double time_to_live{30.0};
    double t0;

    /****************************************************************/
    string getNameFromId(const int id) const
    {
        ostringstream ss;
        ss<<"#"<<hex<<id;
        return ss.str();
    }

    /****************************************************************/
    shared_ptr<MetaSkeleton> create(Bottle *keys)
    {
        shared_ptr<MetaSkeleton> s(new MetaSkeleton(CamParamsHelper(depth.width(),depth.height(),fov_h),
                                                    time_to_live,filter_keypoint_order,filter_limblength_order,
                                                    optimize_limblength));
        vector<pair<string,pair<Vector,Vector>>> unordered;
        auto foot_left=make_pair(string(""),make_pair(Vector(1),Vector(1)));
        auto foot_right=foot_left;
        vector<Vector> shoulders, hips;
        bool shoulder_center_detected=false;
        bool hip_center_detected=false;

        Vector p,pixel(2);
        for (size_t i=0; i<keys->size(); i++)
        {
            if (Bottle *k=keys->get(i).asList())
            {
                if (k->size()==4)
                {
                    string tag=k->get(0).asString();
                    int u=(int)k->get(1).asDouble();
                    int v=(int)k->get(2).asDouble();
                    double confidence=k->get(3).asDouble();

                    if ((confidence>=keys_recognition_confidence) && getPoint3D(u,v,p))
                    {
                        pixel[0]=u; pixel[1]=v;
                        auto pair_=make_pair(keysRemap[tag],make_pair(p,pixel));

                        // handle foot as big-toe; if big-toe is not available,
                        // then use small-toe as fallback
                        if (keysRemap[tag]==KeyPointTag::foot_left)
                        {
                            if (foot_left.first.empty() || (tag=="LBigToe"))
                            {
                                foot_left=pair_;
                            }
                        }
                        else if (keysRemap[tag]==KeyPointTag::foot_right)
                        {
                            if (foot_right.first.empty() || (tag=="RBigToe"))
                            {
                                foot_right=pair_;
                            }
                        }
                        else
                        {
                            unordered.push_back(pair_);
                        }

                        // if [shoulder|hip]_center is not available, then
                        // use (left + right)/2 as fallback
                        if (keysRemap[tag]==KeyPointTag::shoulder_center)
                        {
                            shoulder_center_detected=true;
                            s->pivots[0]=pixel;
                        }
                        else if ((keysRemap[tag]==KeyPointTag::shoulder_left) ||
                                 (keysRemap[tag]==KeyPointTag::shoulder_right))
                        {
                            shoulders.push_back(p);
                        }
                        else if (keysRemap[tag]==KeyPointTag::hip_center)
                        {
                            hip_center_detected=true;
                            s->pivots[1]=pixel;
                        }
                        else if ((keysRemap[tag]==KeyPointTag::hip_left) ||
                                 (keysRemap[tag]==KeyPointTag::hip_right))
                        {
                            hips.push_back(p);
                        }
                    }
                }
                else if (k->size()==3)
                {
                    string tag=k->get(0).asString();
                    string name=k->get(1).asString();
                    double confidence=k->get(2).asDouble();

                    if (tag=="Name")
                    {
                        s->skeleton->setTag(name);
                        s->name_confidence=confidence;
                    }
                }
            }
        }

        if (!foot_left.first.empty())
        {
            unordered.push_back(foot_left);
        }
        if (!foot_right.first.empty())
        {
            unordered.push_back(foot_right);
        }
        if (!shoulder_center_detected && (shoulders.size()==2))
        {
            pixel=numeric_limits<double>::quiet_NaN();
            unordered.push_back(make_pair(KeyPointTag::shoulder_center,
                                          make_pair(0.5*(shoulders[0]+shoulders[1]),pixel)));
        }
        if (!hip_center_detected && (hips.size()==2))
        {
            pixel=numeric_limits<double>::quiet_NaN();
            unordered.push_back(make_pair(KeyPointTag::hip_center,
                                          make_pair(0.5*(hips[0]+hips[1]),pixel)));
        }

        s->skeleton->update_withpixels(unordered);
        return s;
    }

    /****************************************************************/
    void update(const shared_ptr<MetaSkeleton> &src, shared_ptr<MetaSkeleton> &dest,
                vector<string> &remove_tags)
    {
        vector<pair<string,pair<Vector,Vector>>> unordered;
        for (unsigned int i=0; i<src->skeleton->getNumKeyPoints(); i++)
        {
            auto key=(*src->skeleton)[i];
            if (key->isUpdated())
            {
                const Vector &p=key->getPoint();
                unordered.push_back(make_pair(key->getTag(),make_pair(p,key->getPixel())));
                if (dest->keys_acceptable_misses[i]==0)
                    dest->init(key->getTag(),p);
                dest->keys_acceptable_misses[i]=keys_acceptable_misses;
            }
            else if (dest->keys_acceptable_misses[i]>0)
            {
                unordered.push_back(make_pair(key->getTag(),make_pair((*dest->skeleton)[i]->getPoint(),(*dest->skeleton)[i]->getPixel())));
                dest->keys_acceptable_misses[i]--;
            }
        }

        dest->update(unordered);
        dest->timer=time_to_live;
        dest->pivots=src->pivots;

        string oldTag=dest->skeleton->getTag();
        dest->skeleton->setTag(is_unknown(src->skeleton->getTag())?
                               getNameFromId(dest->opc_id):
                               src->skeleton->getTag());

        if (oldTag!=dest->skeleton->getTag())
        {
            remove_tags.push_back(oldTag);
        }
    }

    /****************************************************************/
    bool isValid(const shared_ptr<MetaSkeleton> &s) const
    {
        bool no_pivots=true;
        for (auto &pivot:s->pivots)
        {
            no_pivots=no_pivots && (norm(pivot)==numeric_limits<double>::infinity());
        }
        if (no_pivots)
        {
            return false;
        }

        unsigned int n=0;
        for (unsigned int i=0; i<s->skeleton->getNumKeyPoints(); i++)
        {
            if ((*s->skeleton)[i]->isUpdated())
            {
                n++;
            }
        }
        
        double perc=((double)n)/((double)s->skeleton->getNumKeyPoints());
        double max_path=s->skeleton->getMaxPath();
        return ((perc>=keys_recognition_percentage) && (max_path>=min_acceptable_path));
    }

    /****************************************************************/
    bool opcAdd(shared_ptr<MetaSkeleton> &s, const Stamp &stamp)
    {
        if (opcPort.getOutputCount())
        {
            Bottle cmd,rep;
            cmd.addVocab(Vocab::encode("add"));
            Property prop=applyTransform(s->skeleton)->toProperty();
            if (stamp.isValid())
            {
                prop.put("stamp",stamp.getTime());
            }
            cmd.addList().read(prop);
            if (opcPort.write(cmd,rep))
            {
                if (rep.get(0).asVocab()==Vocab::encode("ack"))
                {
                    s->opc_id=rep.get(1).asList()->get(1).asInt();
                    if (is_unknown(s->skeleton->getTag()))
                    {
                        s->skeleton->setTag(getNameFromId(s->opc_id));
                        return opcSet(s,stamp);
                    }
                    return true;
                }
            }
        }

        return false;
    }

    /****************************************************************/
    bool opcSet(const shared_ptr<MetaSkeleton> &s, const Stamp &stamp)
    {
        if (opcPort.getOutputCount())
        {
            Bottle cmd,rep;
            cmd.addVocab(Vocab::encode("set"));
            Bottle &pl=cmd.addList();
            Property prop=applyTransform(s->skeleton)->toProperty();
            if (stamp.isValid())
            {
                prop.put("stamp",stamp.getTime());
            }
            pl.read(prop);
            Bottle id;
            Bottle &id_pl=id.addList();
            id_pl.addString("id");
            id_pl.addInt(s->opc_id);
            pl.append(id);
            if (opcPort.write(cmd,rep))
            {
                return (rep.get(0).asVocab()==Vocab::encode("ack"));
            }
        }

        return false;
    }

    /****************************************************************/
    bool opcDel(const shared_ptr<MetaSkeleton> &s)
    {
        if (opcPort.getOutputCount())
        {
            Bottle cmd,rep;
            cmd.addVocab(Vocab::encode("del"));
            Bottle &pl=cmd.addList().addList();
            pl.addString("id");
            pl.addInt(s->opc_id);
            if (opcPort.write(cmd,rep))
            {
                return (rep.get(0).asVocab()==Vocab::encode("ack"));
            }
        }

        return false;
    }

    /****************************************************************/
    void gc(const double dt)
    {
        vector<shared_ptr<MetaSkeleton>> skeletons_;
        for (auto &s:skeletons)
        {
            s->timer-=dt;
            if (s->timer>0.0)
            {
                skeletons_.push_back(s);
            }
            else
            {
                opcDel(s);
            }
        }

        skeletons=skeletons_;
    }

    /****************************************************************/
    void viewerUpdate()
    {
        if (viewerPort.getOutputCount()>0)
        {
            Bottle &msg=viewerPort.prepare();
            msg.clear();
            for (auto &s:skeletons)
            {
                Property prop=applyTransform(s->skeleton)->toProperty();
                msg.addList().read(prop);
            }
            viewerPort.writeStrict();
        }
    }

    /****************************************************************/
    bool configure(ResourceFinder &rf) override
    {
        // set up keypoints remapping
        keypoints_map["shoulder_center"]=KeyPointTag::shoulder_center;
        keypoints_map["head"]=KeyPointTag::head;
        keypoints_map["shoulder_left"]=KeyPointTag::shoulder_left;
        keypoints_map["elbow_left"]=KeyPointTag::elbow_left;
        keypoints_map["hand_left"]=KeyPointTag::hand_left;
        keypoints_map["elbow_right"]=KeyPointTag::elbow_right;
        keypoints_map["hand_right"]=KeyPointTag::hand_right;
        keypoints_map["hip_center"]=KeyPointTag::hip_center;
        keypoints_map["hip_left"]=KeyPointTag::hip_left;
        keypoints_map["knee_left"]=KeyPointTag::knee_left;
        keypoints_map["ankle_left"]=KeyPointTag::ankle_left;
        keypoints_map["foot_left"]=KeyPointTag::foot_left;
        keypoints_map["hip_right"]=KeyPointTag::hip_right;
        keypoints_map["knee_right"]=KeyPointTag::knee_right;
        keypoints_map["ankle_right"]=KeyPointTag::ankle_right;
        keypoints_map["foot_right"]=KeyPointTag::foot_right;

        // retrieve values from config file
        Bottle &gGeneral=rf.findGroup("general");
        if (!gGeneral.isNull())
        {
            period=gGeneral.check("period",Value(period)).asDouble();
        }

        Bottle &gSkeleton=rf.findGroup("skeleton");
        if (!gSkeleton.isNull())
        {
            tracking_threshold=gSkeleton.check("tracking-threshold",Value(tracking_threshold)).asDouble();
            time_to_live=gSkeleton.check("time-to-live",Value(time_to_live)).asDouble();

            if (Bottle *b=gSkeleton.check("tracking-keypoints").asList())
            {
                tracking_keypoints.clear();
                for (size_t i=0; i<b->size(); i++)
                {
                    auto &keypoint_tag_it=keypoints_map.find(b->get(i).asString());
                    if (keypoint_tag_it!=end(keypoints_map))
                    {
                        tracking_keypoints.insert(keypoint_tag_it->second);
                    }
                }
            }
        }

        if (tracking_keypoints.empty())
        {
            yError()<<"Specified empty set of keypoints to track!";
            return false;
        }

        skeletonsPort.open("/skeletonRetriever/skeletons:i");
        viewerPort.open("/skeletonRetriever/viewer:o");
        opcPort.open("/skeletonRetriever/opc:rpc");

        t0=Time::now();
        return true;
    }

    /****************************************************************/
    double getPeriod() override
    {
        return period;
    }

    /****************************************************************/
    bool updateModule() override
    {
        const double t=Time::now();
        const double dt=t-t0;
        t0=t;

        if (ImageOf<PixelFloat> *depth=depthPort.read(false))
        {
            if (depth_enable)
            {
                filterDepth(*depth,this->depth,depth_kernel_size,depth_iterations,
                            depth_min_distance,depth_max_distance);
            }
            else
            {
                this->depth=*depth;
            }
        }

        if (!camera_configured)
        {
            camera_configured=getCameraOptions();
        }

        // garbage collector
        gc(dt);

        // update external frames
        update_nav_frame();
        update_gaze_frame();
        rootFrame=navFrame*gazeFrame;

        // handle skeletons acquired from detector
        if (Bottle *b1=skeletonsPort.read(false))
        {
            if (Bottle *b2=b1->get(0).asList())
            {
                // acquire skeletons with sufficient number of key-points
                vector<shared_ptr<MetaSkeleton>> new_accepted_skeletons;
                for (size_t i=0; i<b2->size(); i++)
                {
                    Bottle *b3=b2->get(i).asList();
                    if ((depth.width()>0) && (depth.height()>0) && (b3!=nullptr))
                    {
                        shared_ptr<MetaSkeleton> s=create(b3);
                        if (isValid(s))
                        {
                            new_accepted_skeletons.push_back(s);
                        }
                    }
                }

                // update existing skeletons / create new skeletons
                if (!new_accepted_skeletons.empty())
                {
                    Stamp stamp;
                    skeletonsPort.getEnvelope(stamp);

                    enforce_tag_uniqueness_input(new_accepted_skeletons);

                    vector<shared_ptr<MetaSkeleton>> pending=skeletons;
                    for (auto &n:new_accepted_skeletons)
                    {
                        vector<double> scores=computeScores(pending,n);
                        auto it=min_element(scores.begin(),scores.end());
                        if (it!=scores.end())
                        {
                            if (*it<numeric_limits<double>::infinity())
                            {
                                auto i=distance(scores.begin(),it);
                                auto &s=pending[i];
                                update(n,s,viewer_remove_tags);
                                opcSet(s,stamp);
                                pending.erase(pending.begin()+i);
                                continue;
                            }
                        }

                        if (opcAdd(n,stamp))
                        {
                            skeletons.push_back(n);
                        }
                    }

                    viewerUpdate();
                }
            }
        }

        return true;
    }

    /****************************************************************/
    bool close() override
    {
        // remove all skeletons from OPC
        gc(numeric_limits<double>::infinity());

        skeletonsPort.close();
        viewerPort.close();
        opcPort.close();

        return true;
    }
};


/****************************************************************/
int main(int argc, char *argv[])
{
    Network yarp;
    if (!yarp.checkNetwork())
    {
        yError()<<"Unable to find Yarp server!";
        return EXIT_FAILURE;
    }

    ResourceFinder rf;
    rf.setDefaultContext("skeletonLocker");
    rf.setDefaultConfigFile("config.ini");
    rf.configure(argc,argv);

    Locker locker;
    return locker.runModule(rf);
}

